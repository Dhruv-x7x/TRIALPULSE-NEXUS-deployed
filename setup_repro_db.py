"""
TrialPulse Nexus - Database Setup Script
=========================================
Auto-detects PostgreSQL installation and reads credentials from .env
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Reading environment variables directly.")

# Configuration - Read from .env or use defaults
SOURCE_DUMP = "database/reproduction_dump.sql"
TARGET_DB = os.getenv("DB_NAME", "trialpulse_test")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")


def find_psql():
    """Auto-detect psql.exe by searching common PostgreSQL installation paths."""
    
    # Common installation paths on Windows
    base_paths = [
        r"C:\Program Files\PostgreSQL",
        r"C:\Program Files (x86)\PostgreSQL",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\PostgreSQL"),
    ]
    
    # Check versions from newest to oldest
    versions = ["18", "17", "16", "15", "14", "13", "12"]
    
    for base in base_paths:
        for version in versions:
            psql_path = Path(base) / version / "bin" / "psql.exe"
            if psql_path.exists():
                print(f"✓ Found PostgreSQL {version} at: {psql_path}")
                return str(psql_path)
    
    # Check if psql is in PATH
    try:
        result = subprocess.run(["where", "psql"], capture_output=True, text=True)
        if result.returncode == 0:
            psql_path = result.stdout.strip().split('\n')[0]
            print(f"✓ Found psql in PATH: {psql_path}")
            return psql_path
    except Exception:
        pass
    
    return None


def setup_test_db():
    """Main setup function."""
    
    print("=" * 60)
    print("  TrialPulse Nexus - Database Setup")
    print("=" * 60)
    
    # Check for dump file
    if not os.path.exists(SOURCE_DUMP):
        print(f"\n❌ ERROR: Dump file '{SOURCE_DUMP}' not found.")
        print("   Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Auto-detect PostgreSQL
    print("\n🔍 Searching for PostgreSQL installation...")
    psql_path = find_psql()
    
    if not psql_path:
        print("\n❌ ERROR: Could not find psql.exe")
        print("   Please ensure PostgreSQL is installed.")
        print("   Download from: https://www.postgresql.org/download/windows/")
        sys.exit(1)
    
    # Check for password
    if not DB_PASSWORD:
        print("\n❌ ERROR: DB_PASSWORD not set in .env file")
        print("   Please add your PostgreSQL password to the .env file:")
        print("   DB_PASSWORD=your_password_here")
        sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Database: {TARGET_DB}")
    print(f"   User: {DB_USER}")
    print(f"   Host: {DB_HOST}:{DB_PORT}")
    print(f"   Dump file: {SOURCE_DUMP}")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD

    # 1. Terminate existing connections
    terminate_sql = f"""
    SELECT pg_terminate_backend(pg_stat_activity.pid)
    FROM pg_stat_activity
    WHERE pg_stat_activity.datname = '{TARGET_DB}'
      AND pid <> pg_backend_pid();
    """
    
    print(f"\n🔄 Terminating existing connections to {TARGET_DB}...")
    subprocess.run(
        [psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER, "-d", "postgres", "-c", terminate_sql],
        env=env, stderr=subprocess.DEVNULL
    )

    # 2. Drop the database if it exists
    print(f"🗑️  Dropping existing {TARGET_DB} (if any)...")
    subprocess.run(
        [psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER, "-d", "postgres", "-c", f"DROP DATABASE IF EXISTS {TARGET_DB};"],
        env=env, check=True
    )
    
    # 3. Create the database
    print(f"📦 Creating empty {TARGET_DB}...")
    subprocess.run(
        [psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER, "-d", "postgres", "-c", f"CREATE DATABASE {TARGET_DB};"],
        env=env, check=True
    )

    # 4. Restore the dump
    print(f"📥 Restoring dump into {TARGET_DB}... (this may take 1-2 minutes)")
    cmd_restore = [
        psql_path,
        "-h", DB_HOST,
        "-p", DB_PORT,
        "-U", DB_USER,
        "-d", TARGET_DB,
        "-f", SOURCE_DUMP,
        "-q"  # Quiet mode
    ]
    
    try:
        start_time = time.time()
        subprocess.run(cmd_restore, env=env, check=True)
        duration = time.time() - start_time
        print(f"\n✅ SUCCESS: Database restored in {duration:.2f} seconds!")
        print(f"\n🚀 Next steps:")
        print(f"   1. cd backend")
        print(f"   2. pip install -r requirements.txt")
        print(f"   3. python -m uvicorn app.main:app --reload")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Failed to restore database.")
        print(f"   {e}")
        print(f"\n   Common issues:")
        print(f"   - Wrong password in .env")
        print(f"   - PostgreSQL service not running")
        sys.exit(1)


if __name__ == "__main__":
    setup_test_db()
