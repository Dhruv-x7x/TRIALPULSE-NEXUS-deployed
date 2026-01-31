// User types
export interface User {
  user_id: string;
  username: string;
  email: string | null;
  full_name: string | null;
  role: UserRole;
  permissions: string[];
}

export type UserRole = 'lead' | 'dm' | 'cra' | 'coder' | 'safety' | 'executive';

// Auth types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

// Patient types
export interface Patient {
  patient_key: string;
  site_id: string;
  study_id?: string;
  status?: string;
  risk_level?: string;
  dqi_score?: number;
  clean_status_tier?: string;
  is_db_lock_ready?: boolean;
  enrollment_date?: string;
  last_visit_date?: string;
  open_queries_count?: number;
  open_issues_count?: number;
  visit_compliance_pct?: number;
}

export interface PatientListResponse {
  patients: Patient[];
  total: number;
  page: number;
  page_size: number;
}

// Site types
export interface Site {
  site_id: string;
  name?: string;
  country?: string;
  region?: string;
  status?: string;
  performance_score?: number;
  dqi_score?: number;
  patient_count?: number;
}

export interface SiteBenchmark {
  site_id: string;
  patient_count: number;
  dqi_score: number;
  visit_compliance: number;
  query_rate: number;
  enrollment_rate: number;
  cascade_mean: number;
  avg_risk_score: number;
}

// Study types
export interface Study {
  study_id: string;
  name?: string;
  protocol_number?: string;
  phase?: string;
  status?: string;
  therapeutic_area?: string;
  sponsor?: string;
  target_enrollment?: number;
  current_enrollment?: number;
}

// Issue types
export interface Issue {
  issue_id?: number;
  patient_key?: string;
  site_id?: string;
  issue_type?: string;
  priority?: 'Critical' | 'High' | 'Medium' | 'Low';
  status?: string;
  description?: string;
  created_at?: string;
}

export interface IssueSummary {
  total: number;
  by_status: Record<string, number>;
  by_priority: Record<string, number>;
  open_count: number;
  critical_count: number;
}

// Analytics types
export interface PortfolioSummary {
  total_patients: number;
  total_sites: number;
  total_studies: number;
  total_issues: number;
  mean_dqi: number;
  dblock_ready_count: number;
  dblock_ready_rate: number;
  tier2_clean_count: number;
  tier2_clean_rate: number;
  critical_issues: number;
  high_issues: number;
}

export interface DQIDistribution {
  dqi_band: string;
  count: number;
}

export interface RegionalMetric {
  region: string;
  site_count: number;
  avg_dqi: number;
  avg_performance: number;
}

// ML types
export interface MLModel {
  version_id?: number;
  model_name: string;
  model_type?: string;
  version?: string;
  status?: string;
  trained_at?: string;
  deployed_at?: string;
  training_samples?: number;
}

// Report types
export type ReportType = 
  | 'cra_monitoring'
  | 'site_performance'
  | 'executive_brief'
  | 'db_lock_readiness'
  | 'daily_digest'
  | 'query_summary'
  | 'sponsor_update'
  | 'meeting_pack'
  | 'safety_narrative'
  | 'inspection_prep'
  | 'site_newsletter'
  | 'issue_escalation';

export interface ReportTypeInfo {
  id: ReportType;
  name: string;
  description: string;
}

export interface ReportRequest {
  report_type: ReportType;
  site_id?: string;
  study_id?: string;
  date_range_days?: number;
  format?: 'html' | 'pdf';
}

export interface ReportResponse {
  report_type: string;
  generated_at: string;
  content: string;
  format: string;
  metadata: Record<string, unknown>;
}

// Cascade types
export interface CascadeImpact {
  patient_key: string;
  site_id: string;
  cascade_impact_score: number;
  blocking_issues: number;
  open_queries_count: number;
  dqi_score: number;
}

// Pattern Alert types
export interface PatternAlert {
  pattern_id: string;
  pattern_name: string;
  severity: string;
  match_count: number;
  sites_affected: number;
  last_detected?: string;
  status: string;
  alert_message?: string;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  total?: number;
  page?: number;
  page_size?: number;
}

// Table Column types
export interface TableColumn<T> {
  key: keyof T | string;
  header: string;
  sortable?: boolean;
  render?: (value: unknown, row: T) => React.ReactNode;
}
