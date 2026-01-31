@echo off
echo ==================================================
echo TRIAL PULSE NEXUS - NEO4J CASCADE SEEDER
echo ==================================================
echo.

IF NOT EXIST venv (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please ensure you are in the project root directory.
    pause
    exit /b
)

echo [1/2] Activating virtual environment...
call venv\Scripts\activate

echo [2/2] Running Neo4j Cascade Seeding script...
python scripts\seed_neo4j_cascade.py

echo.
echo ==================================================
echo Seeding Process Complete!
echo ==================================================
pause
