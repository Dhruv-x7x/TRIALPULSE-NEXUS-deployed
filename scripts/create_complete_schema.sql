-- ============================================================================
-- TRIALPULSE NEXUS - Complete Database Schema Fix
-- Creates all missing tables and views for full functionality
-- ============================================================================

-- 0. Ensure Unified Patient Record View exists first (since others depend on it)
-- This view maps the primary 'patients' table to the expected legacy UPR format
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'unified_patient_record') THEN
        -- We check if 'patients' table exists first to avoid errors
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'patients') THEN
            CREATE VIEW unified_patient_record AS 
            SELECT 
                patient_key, study_id, site_id, status, risk_level, risk_score, 
                dqi_score, clean_status_tier, is_db_lock_ready,
                (open_queries_count > 0) as has_open_queries,
                (open_issues_count > 0) as has_missing_pages,
                (all_signatures_complete IS FALSE) as has_signature_gaps,
                (visit_compliance_pct < 100) as has_sdv_incomplete
            FROM patients;
        END IF;
    END IF;
END $$;

-- ============================================================================
-- PATTERN ALERTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS pattern_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    pattern_id VARCHAR(64) NOT NULL,
    pattern_name VARCHAR(256),
    site_id VARCHAR(64),
    study_id VARCHAR(64),
    patient_key VARCHAR(64),
    severity VARCHAR(32) DEFAULT 'medium',
    alert_message TEXT,
    confidence DECIMAL(5,4),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(32) DEFAULT 'active',
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(64),
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_pattern_alerts_site ON pattern_alerts(site_id);
CREATE INDEX IF NOT EXISTS idx_pattern_alerts_severity ON pattern_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_pattern_alerts_status ON pattern_alerts(status);

-- ============================================================================
-- PATTERN LIBRARY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS pattern_library (
    pattern_id VARCHAR(64) PRIMARY KEY,
    pattern_name VARCHAR(256) NOT NULL,
    pattern_type VARCHAR(64),
    description TEXT,
    trigger_conditions JSONB,
    expected_impact JSONB,
    success_rate DECIMAL(5,4),
    sample_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB
);

-- ============================================================================
-- COLLABORATION TABLES
-- ============================================================================

-- Investigation Rooms
CREATE TABLE IF NOT EXISTS investigation_rooms (
    room_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(256) NOT NULL,
    room_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    description TEXT,
    created_by VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    context JSONB
);

CREATE TABLE IF NOT EXISTS room_participants (
    participant_id VARCHAR(64) PRIMARY KEY,
    room_id VARCHAR(64) REFERENCES investigation_rooms(room_id),
    user_id VARCHAR(64) NOT NULL,
    user_name VARCHAR(128),
    role VARCHAR(32),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS room_evidence (
    evidence_id VARCHAR(64) PRIMARY KEY,
    room_id VARCHAR(64) REFERENCES investigation_rooms(room_id),
    evidence_type VARCHAR(32),
    title VARCHAR(256),
    content TEXT,
    strength VARCHAR(32),
    submitted_by VARCHAR(64),
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS room_threads (
    thread_id VARCHAR(64) PRIMARY KEY,
    room_id VARCHAR(64) REFERENCES investigation_rooms(room_id),
    title VARCHAR(256),
    thread_type VARCHAR(32),
    created_by VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Team Workspaces
CREATE TABLE IF NOT EXISTS team_workspaces (
    workspace_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    workspace_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) DEFAULT 'active',
    description TEXT,
    created_by VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workspace_members (
    member_id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) REFERENCES team_workspaces(workspace_id),
    user_id VARCHAR(64) NOT NULL,
    user_name VARCHAR(128),
    role VARCHAR(32),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workspace_goals (
    goal_id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) REFERENCES team_workspaces(workspace_id),
    title VARCHAR(256),
    description TEXT,
    status VARCHAR(32) DEFAULT 'pending',
    priority VARCHAR(32),
    due_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Issue Comments (issues table already exists but might need enhancement)
CREATE TABLE IF NOT EXISTS issue_comments (
    comment_id VARCHAR(64) PRIMARY KEY,
    issue_id VARCHAR(64),
    content TEXT,
    author VARCHAR(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Escalations
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id VARCHAR(64) PRIMARY KEY,
    issue_id VARCHAR(64),
    level INTEGER DEFAULT 1,
    status VARCHAR(32) DEFAULT 'active',
    title VARCHAR(256),
    assigned_to VARCHAR(128),
    sla_deadline TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- User Alerts
CREATE TABLE IF NOT EXISTS user_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    recipient_id VARCHAR(64) NOT NULL,
    title VARCHAR(256) NOT NULL,
    message TEXT,
    category VARCHAR(32),
    priority VARCHAR(32),
    status VARCHAR(32) DEFAULT 'unread',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    read_at TIMESTAMP WITH TIME ZONE
);

-- Tags
CREATE TABLE IF NOT EXISTS tags (
    tag_id VARCHAR(64) PRIMARY KEY,
    tag_type VARCHAR(32),
    tag_value VARCHAR(256),
    entity_type VARCHAR(64),
    entity_id VARCHAR(64),
    created_by VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for collaboration tables
CREATE INDEX IF NOT EXISTS idx_rooms_status ON investigation_rooms(status);
CREATE INDEX IF NOT EXISTS idx_escalations_level ON escalations(level);
CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);
CREATE INDEX IF NOT EXISTS idx_alerts_recipient ON user_alerts(recipient_id, status);

-- ============================================================================
-- PATIENT ANALYTICS VIEWS (if not exist as tables)
-- ============================================================================

-- Check if patient_dqi exists, if not create a view
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'patient_dqi') THEN
        CREATE VIEW patient_dqi AS
        SELECT 
            patient_key,
            study_id,
            site_id,
            COALESCE(dqi_score, 85.0) as enhanced_dqi,
            COALESCE(dqi_score, 85.0) as dqi_score,
            'stable' as dqi_trend
        FROM unified_patient_record;
    END IF;
END $$;

-- Check if patient_clean_status exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'patient_clean_status') THEN
        CREATE VIEW patient_clean_status AS
        SELECT 
            patient_key,
            study_id,
            site_id,
            CASE 
                WHEN dqi_score >= 95 THEN 'db_lock_ready'
                WHEN dqi_score >= 85 THEN 'tier_2'
                WHEN dqi_score >= 70 THEN 'tier_1'
                ELSE 'tier_0'
            END as clean_status_tier,
            CASE WHEN dqi_score >= 85 THEN TRUE ELSE FALSE END as tier2_clean,
            CASE WHEN dqi_score >= 95 THEN TRUE ELSE FALSE END as db_lock_tier1_ready
        FROM unified_patient_record;
    END IF;
END $$;

-- Check if patient_dblock_status exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'patient_dblock_status') THEN
        CREATE VIEW patient_dblock_status AS
        SELECT 
            patient_key,
            study_id,
            site_id,
            CASE 
                WHEN dqi_score >= 95 THEN 'Tier 1 - Ready'
                WHEN dqi_score >= 85 THEN 'Tier 2 - Near Ready'
                WHEN dqi_score >= 70 THEN 'In Progress'
                ELSE 'Not Ready'
            END as db_lock_status,
            CASE WHEN dqi_score >= 95 THEN TRUE ELSE FALSE END as db_lock_tier1_ready,
            CASE WHEN dqi_score >= 85 THEN TRUE ELSE FALSE END as tier2_clean,
            dqi_score as enhanced_dqi
        FROM unified_patient_record;
    END IF;
END $$;

-- Check if patient_issues exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'patient_issues') THEN
        CREATE VIEW patient_issues AS
        SELECT 
            patient_key,
            study_id,
            site_id,
            CASE WHEN has_open_queries = TRUE THEN TRUE ELSE FALSE END as issue_open_queries,
            CASE WHEN has_missing_pages = TRUE THEN TRUE ELSE FALSE END as issue_missing_pages,
            CASE WHEN has_signature_gaps = TRUE THEN TRUE ELSE FALSE END as issue_signature_gaps,
            CASE WHEN has_sdv_incomplete = TRUE THEN TRUE ELSE FALSE END as issue_sdv_incomplete,
            FALSE as issue_sae_dm_pending,
            FALSE as issue_sae_safety_pending,
            FALSE as issue_broken_signatures,
            FALSE as issue_meddra_uncoded,
            FALSE as issue_whodrug_uncoded
        FROM unified_patient_record;
    END IF;
END $$;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
