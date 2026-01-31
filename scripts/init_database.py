"""
TRIALPULSE NEXUS - Database Initialization Script
==================================================
Creates all database tables and seeds with initial data.

Usage:
    python scripts/init_database.py [--drop] [--seed]
    
Options:
    --drop  Drop existing tables before creating (DESTRUCTIVE!)
    --seed  Seed with sample data after creating tables
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import random
import uuid
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Base, Study, ClinicalSite, Patient, Visit, LabResult, AdverseEvent,
    ProjectIssue, Query, ResolutionAction, Signature, AuditLog,
    User, Role, MLModelVersion, DriftReport,
    study_sites
)
from src.database.enums import (
    PatientStatus, CleanStatusTier, RiskLevel,
    SiteStatus, Region,
    StudyPhase, StudyStatus,
    VisitStatus, VisitType,
    IssueStatus, IssuePriority, IssueSeverity, IssueCategory,
    QueryStatus, QueryType,
    AdverseEventSeverity, AdverseEventCausality, AdverseEventOutcome, SAEClassification,
    SignatureType, SignatureMeaning,
    AuditAction, EntityType,
    UserRole, UserStatus,
    ModelType, ModelStatus, DriftSeverity,
)
from src.database.config import get_database_url


def get_engine():
    """Create database engine."""
    url = get_database_url()
    return create_engine(url, echo=False)


def create_tables(engine, drop_existing=False):
    """Create all database tables."""
    print("\n" + "="*60)
    print("CREATING DATABASE TABLES")
    print("="*60)
    
    if drop_existing:
        print("\n⚠️  Dropping all existing tables...")
        Base.metadata.drop_all(engine)
        print("   Tables dropped.")
    
    print("\n📋 Creating tables...")
    Base.metadata.create_all(engine)
    
    # List created tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\n✅ Created {len(tables)} tables:")
    for table in sorted(tables):
        print(f"   • {table}")
    
    return tables


def seed_roles(session):
    """Seed default user roles."""
    print("\n📋 Seeding roles...")
    
    roles = [
        Role(
            role_id="system_admin",
            name="System Administrator",
            description="Full system access",
            level=100,
            permissions={"all": True}
        ),
        Role(
            role_id="study_lead",
            name="Study Lead",
            description="Study-level management",
            level=90,
            permissions={"study": ["read", "write"], "patient": ["read", "write"], "site": ["read", "write"]}
        ),
        Role(
            role_id="cra",
            name="Clinical Research Associate",
            description="Site monitoring",
            level=70,
            permissions={"patient": ["read", "write"], "site": ["read"], "issue": ["read", "write"]}
        ),
        Role(
            role_id="data_manager",
            name="Data Manager",
            description="Data entry and query resolution",
            level=60,
            permissions={"patient": ["read", "write"], "query": ["read", "write"]}
        ),
        Role(
            role_id="safety_officer",
            name="Safety Officer",
            description="Safety data access",
            level=80,
            permissions={"adverse_event": ["read", "write"], "patient": ["read"]}
        ),
        Role(
            role_id="medical_coder",
            name="Medical Coder",
            description="Coding workflows",
            level=50,
            permissions={"adverse_event": ["read", "write"]}
        ),
        Role(
            role_id="auditor",
            name="Auditor",
            description="Read-only audit access",
            level=80,
            permissions={"audit_log": ["read"], "patient": ["read"], "study": ["read"]}
        ),
        Role(
            role_id="viewer",
            name="Viewer",
            description="Read-only dashboard access",
            level=10,
            permissions={"dashboard": ["read"]}
        ),
    ]
    
    for role in roles:
        existing = session.query(Role).filter_by(role_id=role.role_id).first()
        if not existing:
            session.add(role)
    
    session.commit()
    print(f"   Created {len(roles)} roles")


def seed_users(session):
    """Seed default admin user."""
    print("\n📋 Seeding users...")
    
    # Hash password
    password_hash = hashlib.sha256("admin123".encode()).hexdigest()
    
    admin = User(
        user_id="admin",
        username="admin",
        email="admin@trialpulse.com",
        password_hash=password_hash,
        first_name="System",
        last_name="Administrator",
        status=UserStatus.ACTIVE.value,
        mfa_enabled=False,
    )
    
    existing = session.query(User).filter_by(user_id="admin").first()
    if not existing:
        session.add(admin)
        session.commit()
        
        # Assign admin role
        admin_role = session.query(Role).filter_by(role_id="system_admin").first()
        if admin_role:
            session.execute(
                text("INSERT INTO user_role_assignments (user_id, role_id, assigned_by) VALUES (:uid, :rid, :by)")
                .bindparams(uid="admin", rid="system_admin", by="system")
            )
            session.commit()
    
    print("   Created admin user")


def seed_studies(session):
    """Seed sample studies."""
    print("\n📋 Seeding studies...")
    
    studies = [
        Study(
            study_id="STUDY-001",
            name="NEXUS-2025 Phase 3 Oncology Trial",
            protocol_number="NEXUS-2025-001",
            phase=StudyPhase.PHASE_III.value,
            status=StudyStatus.ACTIVE.value,
            therapeutic_area="Oncology",
            indication="Non-Small Cell Lung Cancer",
            sponsor="TrialPulse Pharma",
            start_date=datetime(2024, 1, 15),
            target_enrollment=500,
            current_enrollment=347,
        ),
        Study(
            study_id="STUDY-002",
            name="CARDIO-SAFE Cardiovascular Study",
            protocol_number="CARDIO-2024-002",
            phase=StudyPhase.PHASE_II.value,
            status=StudyStatus.ENROLLING.value,
            therapeutic_area="Cardiology",
            indication="Heart Failure",
            sponsor="TrialPulse Pharma",
            start_date=datetime(2024, 6, 1),
            target_enrollment=300,
            current_enrollment=156,
        ),
    ]
    
    for study in studies:
        existing = session.query(Study).filter_by(study_id=study.study_id).first()
        if not existing:
            session.add(study)
    
    session.commit()
    print(f"   Created {len(studies)} studies")


def seed_sites(session):
    """Seed sample clinical sites."""
    print("\n📋 Seeding sites...")
    
    sites_data = [
        # North America
        ("US-001", "Mayo Clinic Rochester", "USA", Region.NORTH_AMERICA.value, "Rochester"),
        ("US-002", "Johns Hopkins Medical", "USA", Region.NORTH_AMERICA.value, "Baltimore"),
        ("US-003", "Stanford Medical Center", "USA", Region.NORTH_AMERICA.value, "Palo Alto"),
        ("CA-001", "Toronto General Hospital", "Canada", Region.NORTH_AMERICA.value, "Toronto"),
        
        # Europe
        ("UK-001", "Royal London Hospital", "UK", Region.EUROPE.value, "London"),
        ("DE-001", "Charité Berlin", "Germany", Region.EUROPE.value, "Berlin"),
        ("FR-001", "Hôpital de la Pitié-Salpêtrière", "France", Region.EUROPE.value, "Paris"),
        ("ES-001", "Hospital Clínic Barcelona", "Spain", Region.EUROPE.value, "Barcelona"),
        
        # Asia Pacific
        ("JP-001", "Tokyo University Hospital", "Japan", Region.ASIA_PACIFIC.value, "Tokyo"),
        ("AU-001", "Royal Melbourne Hospital", "Australia", Region.ASIA_PACIFIC.value, "Melbourne"),
        ("SG-001", "Singapore General Hospital", "Singapore", Region.ASIA_PACIFIC.value, "Singapore"),
        ("KR-001", "Samsung Medical Center", "South Korea", Region.ASIA_PACIFIC.value, "Seoul"),
        
        # Latin America
        ("BR-001", "Hospital Albert Einstein", "Brazil", Region.LATIN_AMERICA.value, "São Paulo"),
        ("MX-001", "Hospital ABC México", "Mexico", Region.LATIN_AMERICA.value, "Mexico City"),
        ("AR-001", "Hospital Italiano", "Argentina", Region.LATIN_AMERICA.value, "Buenos Aires"),
    ]
    
    sites = []
    for site_id, name, country, region, city in sites_data:
        site = ClinicalSite(
            site_id=site_id,
            name=name,
            country=country,
            region=region,
            city=city,
            status=SiteStatus.ACTIVE.value,
            activation_date=datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60)),
            performance_score=random.uniform(70, 98),
            dqi_score=random.uniform(75, 100),
            risk_level=random.choice([RiskLevel.LOW.value, RiskLevel.MEDIUM.value]),
            enrollment_rate=random.uniform(0.5, 2.0),
            query_resolution_days=random.uniform(2, 10),
            principal_investigator=f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
        )
        sites.append(site)
    
    for site in sites:
        existing = session.query(ClinicalSite).filter_by(site_id=site.site_id).first()
        if not existing:
            session.add(site)
    
    session.commit()
    
    # Link sites to studies
    study1 = session.query(Study).filter_by(study_id="STUDY-001").first()
    study2 = session.query(Study).filter_by(study_id="STUDY-002").first()
    
    if study1:
        for site in sites[:10]:
            session.execute(
                text("INSERT INTO study_sites (study_id, site_id, status) VALUES (:sid, :siteid, 'active') ON CONFLICT DO NOTHING")
                .bindparams(sid="STUDY-001", siteid=site.site_id)
            )
    
    if study2:
        for site in sites[5:15]:
            session.execute(
                text("INSERT INTO study_sites (study_id, site_id, status) VALUES (:sid, :siteid, 'active') ON CONFLICT DO NOTHING")
                .bindparams(sid="STUDY-002", siteid=site.site_id)
            )
    
    session.commit()
    print(f"   Created {len(sites)} sites")


def seed_patients(session, count=100):
    """Seed sample patients."""
    print(f"\n📋 Seeding {count} patients...")
    
    sites = session.query(ClinicalSite).all()
    if not sites:
        print("   ⚠️ No sites found, skipping patients")
        return
    
    statuses = [PatientStatus.ACTIVE.value] * 7 + [PatientStatus.COMPLETED.value] * 2 + [PatientStatus.WITHDRAWN.value]
    clean_tiers = [CleanStatusTier.TIER_0.value] * 3 + [CleanStatusTier.TIER_1.value] * 4 + [CleanStatusTier.TIER_2.value] * 2 + [CleanStatusTier.DB_LOCK_READY.value]
    
    patients_created = 0
    for i in range(count):
        site = random.choice(sites)
        patient_key = f"PAT-{i+1:05d}"
        
        existing = session.query(Patient).filter_by(patient_key=patient_key).first()
        if existing:
            continue
        
        status = random.choice(statuses)
        clean_tier = random.choice(clean_tiers)
        has_issues = clean_tier == CleanStatusTier.TIER_0.value
        
        patient = Patient(
            patient_key=patient_key,
            study_id="STUDY-001",
            site_id=site.site_id,
            status=status,
            enrollment_date=datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300)),
            age_at_enrollment=random.randint(25, 75),
            gender=random.choice(["Male", "Female"]),
            clean_status_tier=clean_tier,
            is_db_lock_ready=clean_tier == CleanStatusTier.DB_LOCK_READY.value,
            risk_level=random.choice([RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]),
            risk_score=random.uniform(0, 1),
            dqi_score=random.uniform(60, 100),
            has_issues=has_issues,
            open_issues_count=random.randint(0, 5) if has_issues else 0,
            open_queries_count=random.randint(0, 8) if has_issues else 0,
            total_visits_planned=8,
            total_visits_completed=random.randint(1, 8),
            visit_compliance_pct=random.uniform(70, 100),
            all_signatures_complete=not has_issues,
            consent_valid=True,
            has_sae=random.random() < 0.05,  # 5% SAE rate
            sae_count=1 if random.random() < 0.05 else 0,
            days_since_last_activity=random.randint(0, 30),
            data_entry_lag_days=random.uniform(0, 7),
            avg_query_age_days=random.uniform(0, 15),
        )
        
        session.add(patient)
        patients_created += 1
    
    session.commit()
    print(f"   Created {patients_created} patients")


def seed_visits(session):
    """Seed patient visits."""
    print("\n📋 Seeding visits...")
    
    patients = session.query(Patient).limit(50).all()
    visits_created = 0
    
    visit_schedule = [
        ("V1", "Screening", VisitType.SCREENING.value, 0),
        ("V2", "Baseline", VisitType.BASELINE.value, 7),
        ("V3", "Week 2", VisitType.TREATMENT.value, 14),
        ("V4", "Week 4", VisitType.TREATMENT.value, 28),
        ("V5", "Week 8", VisitType.TREATMENT.value, 56),
        ("V6", "Week 12", VisitType.TREATMENT.value, 84),
        ("V7", "Week 24", VisitType.FOLLOW_UP.value, 168),
        ("V8", "End of Study", VisitType.END_OF_STUDY.value, 196),
    ]
    
    for patient in patients:
        completed_visits = patient.total_visits_completed
        
        for i, (visit_num, visit_name, visit_type, day_offset) in enumerate(visit_schedule):
            scheduled_date = patient.enrollment_date + timedelta(days=day_offset) if patient.enrollment_date else None
            
            if i < completed_visits:
                status = VisitStatus.COMPLETED.value
                actual_date = scheduled_date + timedelta(days=random.randint(-2, 5)) if scheduled_date else None
            elif i == completed_visits:
                status = random.choice([VisitStatus.SCHEDULED.value, VisitStatus.PARTIALLY_COMPLETED.value])
                actual_date = None
            else:
                status = VisitStatus.SCHEDULED.value
                actual_date = None
            
            visit = Visit(
                visit_id=f"{patient.patient_key}-{visit_num}",
                patient_key=patient.patient_key,
                visit_number=i + 1,
                visit_name=visit_name,
                visit_type=visit_type,
                scheduled_date=scheduled_date,
                actual_date=actual_date,
                status=status,
                is_in_window=True,
                deviation_days=0,
                data_entry_complete=status == VisitStatus.COMPLETED.value,
                sdv_complete=status == VisitStatus.COMPLETED.value and random.random() > 0.3,
                queries_resolved=status == VisitStatus.COMPLETED.value,
            )
            
            existing = session.query(Visit).filter_by(visit_id=visit.visit_id).first()
            if not existing:
                session.add(visit)
                visits_created += 1
    
    session.commit()
    print(f"   Created {visits_created} visits")


def seed_issues(session):
    """Seed project issues."""
    print("\n📋 Seeding issues...")
    
    patients_with_issues = session.query(Patient).filter(Patient.has_issues == True).limit(30).all()
    issues_created = 0
    
    issue_templates = [
        (IssueCategory.MISSING_VISITS.value, "Missing visit data", IssuePriority.HIGH.value),
        (IssueCategory.OVERDUE_QUERIES.value, "Overdue query pending", IssuePriority.MEDIUM.value),
        (IssueCategory.SIGNATURE_GAPS.value, "PI signature missing", IssuePriority.HIGH.value),
        (IssueCategory.PROTOCOL_DEVIATION.value, "Minor protocol deviation", IssuePriority.CRITICAL.value),
        (IssueCategory.LAB_DISCREPANCY.value, "Lab value out of range", IssuePriority.MEDIUM.value),
        (IssueCategory.CONSENT_ISSUE.value, "Consent form version update needed", IssuePriority.HIGH.value),
        (IssueCategory.SDV_INCOMPLETE.value, "SDV pending completion", IssuePriority.LOW.value),
        (IssueCategory.CODING_REQUIRED.value, "MedDRA coding required", IssuePriority.LOW.value),
    ]
    
    for patient in patients_with_issues:
        num_issues = random.randint(1, 3)
        
        for _ in range(num_issues):
            category, desc, priority = random.choice(issue_templates)
            
            issue = ProjectIssue(
                issue_id=str(uuid.uuid4())[:8],
                patient_key=patient.patient_key,
                site_id=patient.site_id,
                category=category,
                issue_type=category,
                description=f"{desc} for patient {patient.patient_key}",
                priority=priority,
                severity=random.choice([IssueSeverity.MINOR.value, IssueSeverity.MAJOR.value]),
                status=random.choice([IssueStatus.OPEN.value, IssueStatus.IN_PROGRESS.value]),
                assigned_to="Data Manager",
                assigned_role=UserRole.DATA_MANAGER.value,
                blocking_count=random.randint(0, 5),
                cascade_impact_score=random.uniform(0, 10),
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                due_date=datetime.utcnow() + timedelta(days=random.randint(1, 14)),
            )
            
            session.add(issue)
            issues_created += 1
    
    session.commit()
    print(f"   Created {issues_created} issues")


def seed_queries(session):
    """Seed data queries."""
    print("\n📋 Seeding queries...")
    
    patients = session.query(Patient).filter(Patient.open_queries_count > 0).limit(20).all()
    queries_created = 0
    
    query_templates = [
        ("Lab Value", "Please verify the ALT value - appears above normal range"),
        ("Visit Date", "Visit date appears outside protocol window"),
        ("Concomitant Medication", "Please clarify medication dose"),
        ("Adverse Event", "Please provide additional details on this AE"),
        ("Demographics", "Please verify patient age"),
    ]
    
    for patient in patients:
        num_queries = min(patient.open_queries_count, 3)
        
        for _ in range(num_queries):
            field, qtext = random.choice(query_templates)
            status = random.choice([QueryStatus.OPEN.value, QueryStatus.OPEN.value, QueryStatus.ANSWERED.value])
            
            query = Query(
                query_id=str(uuid.uuid4())[:8],
                patient_key=patient.patient_key,
                field_name=field,
                form_name="CRF Page",
                query_text=qtext,
                query_type=QueryType.MANUAL.value,
                status=status,
                response_text="Data verified and corrected" if status == QueryStatus.ANSWERED.value else None,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 20)),
                answered_at=datetime.utcnow() if status == QueryStatus.ANSWERED.value else None,
                age_days=random.randint(1, 20),
            )
            
            session.add(query)
            queries_created += 1
    
    session.commit()
    print(f"   Created {queries_created} queries")


def seed_ml_models(session):
    """Seed ML model registry."""
    print("\n📋 Seeding ML model versions...")
    
    models = [
        (ModelType.RISK_CLASSIFIER.value, "risk_classifier", "v9.0", "data/processed/ml/models/"),
        (ModelType.ISSUE_DETECTOR.value, "issue_detector", "v3.0", "models/issue_detector/"),
        (ModelType.SITE_RANKER.value, "site_ranker", "v2.0", "data/processed/ml/site_ranker/"),
        (ModelType.RESOLUTION_PREDICTOR.value, "resolution_predictor", "v3.0", "data/processed/ml/resolution_time/"),
        (ModelType.ANOMALY_DETECTOR.value, "anomaly_detector", "v2.0", "data/processed/ml/anomaly/"),
    ]
    
    for model_type, name, version, path in models:
        model = MLModelVersion(
            version_id=f"{name}-{version}",
            model_name=name,
            model_type=model_type,
            version=version,
            status=ModelStatus.DEPLOYED.value,
            artifact_path=path,
            training_metrics={"accuracy": 0.92, "f1": 0.88, "auc": 0.95},
            training_samples=57000,
            trained_at=datetime.utcnow() - timedelta(days=7),
            deployed_at=datetime.utcnow() - timedelta(days=5),
        )
        
        existing = session.query(MLModelVersion).filter_by(version_id=model.version_id).first()
        if not existing:
            session.add(model)
    
    session.commit()
    print(f"   Created {len(models)} ML model versions")


def seed_audit_log(session, user_id="admin"):
    """Create initial audit entry."""
    print("\n📋 Creating initial audit entry...")
    
    checksum = hashlib.sha256(f"init-{datetime.utcnow().isoformat()}".encode()).hexdigest()
    
    log = AuditLog(
        log_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow(),
        user_id=user_id,
        user_name="System Administrator",
        user_role=UserRole.SYSTEM_ADMIN.value,
        action=AuditAction.CREATE.value,
        entity_type=EntityType.STUDY.value,
        entity_id="system",
        reason="Database initialization",
        checksum=checksum,
        previous_checksum=None,
    )
    
    session.add(log)
    session.commit()
    print("   Created initial audit entry")


def run_init(drop_existing=False, seed_data=True):
    """Run full database initialization."""
    print("\n" + "="*60)
    print("TRIALPULSE NEXUS - DATABASE INITIALIZATION")
    print("="*60)
    
    engine = get_engine()
    
    # Test connection
    print("\n🔗 Testing database connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("   Connection successful!")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Create tables
    tables = create_tables(engine, drop_existing)
    
    if seed_data:
        print("\n" + "="*60)
        print("SEEDING DATABASE")
        print("="*60)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            seed_roles(session)
            seed_users(session)
            seed_studies(session)
            seed_sites(session)
            seed_patients(session, count=100)
            seed_visits(session)
            seed_issues(session)
            seed_queries(session)
            seed_ml_models(session)
            seed_audit_log(session)
            
            # Get counts
            patient_count = session.query(Patient).count()
            site_count = session.query(ClinicalSite).count()
            issue_count = session.query(ProjectIssue).count()
            
            print("\n" + "="*60)
            print("INITIALIZATION COMPLETE")
            print("="*60)
            print(f"\n📊 Database Summary:")
            print(f"   • Tables: {len(tables)}")
            print(f"   • Patients: {patient_count}")
            print(f"   • Sites: {site_count}")
            print(f"   • Issues: {issue_count}")
            
        except Exception as e:
            session.rollback()
            print(f"\n❌ Error during seeding: {e}")
            raise
        finally:
            session.close()
    
    print("\n✅ Database ready!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Initialize TrialPulse Nexus database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables (DESTRUCTIVE)")
    parser.add_argument("--seed", action="store_true", default=True, help="Seed with sample data")
    parser.add_argument("--no-seed", action="store_true", help="Skip seeding data")
    
    args = parser.parse_args()
    
    seed_data = not args.no_seed
    
    if args.drop:
        confirm = input("\n⚠️  This will DELETE all existing data. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
    
    success = run_init(drop_existing=args.drop, seed_data=seed_data)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
