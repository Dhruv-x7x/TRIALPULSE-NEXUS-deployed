
import os
import sys
import psycopg2
from psycopg2 import sql

# Configuration
ORIGINAL_DB = "trialpulse_nexus"
TEST_DB = "trialpulse_test"
DB_USER = "postgres"
DB_PASSWORD = "chitti"
DB_HOST = "127.0.0.1"

def get_db_connection(db_name):
    return psycopg2.connect(
        dbname=db_name,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST
    )

def get_table_counts(conn):
    cursor = conn.cursor()
    # Get all tables in public schema
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    counts = {}
    for table in tables:
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
        count = cursor.fetchone()[0]
        counts[table] = count
    
    return counts

def verify_databases():
    print(f"Verifying match between '{ORIGINAL_DB}' and '{TEST_DB}'...")
    
    try:
        conn_orig = get_db_connection(ORIGINAL_DB)
        conn_test = get_db_connection(TEST_DB)
        
        print("Connected to both databases.")
        
        counts_orig = get_table_counts(conn_orig)
        counts_test = get_table_counts(conn_test)
        
        conn_orig.close()
        conn_test.close()
        
        # Compare
        all_tables = set(counts_orig.keys()) | set(counts_test.keys())
        mismatches = []
        
        print(f"\nComparing {len(all_tables)} tables...")
        
        for table in sorted(all_tables):
            count_orig = counts_orig.get(table, "MISSING")
            count_test = counts_test.get(table, "MISSING")
            
            if count_orig != count_test:
                mismatches.append((table, count_orig, count_test))
                print(f"MISMATCH: {table} (Orig: {count_orig}, Test: {count_test})")
        
        if not mismatches:
            print("\nSUCCESS: Both databases have exactly the same tables and row counts!")
            print("The reproduction process is verified.")
        else:
            print(f"\nFAILURE: Found {len(mismatches)} mismatches.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_databases()
