# Save as: src/generation/template_engine.py

"""
TRIALPULSE NEXUS 10X - Template Engine v1.0
Jinja2-based template system with multi-format support
Hardened for high-contrast production reporting.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from jinja2 import Environment, FileSystemLoader, select_autoescape, BaseLoader, TemplateNotFound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """12 supported report types"""
    CRA_MONITORING = "cra_monitoring"
    SITE_PERFORMANCE = "site_performance"
    SPONSOR_UPDATE = "sponsor_update"
    MEETING_PACK = "meeting_pack"
    SAFETY_NARRATIVE = "safety_narrative"
    INSPECTION_PREP = "inspection_prep"
    QUERY_SUMMARY = "query_summary"
    SITE_NEWSLETTER = "site_newsletter"
    EXECUTIVE_BRIEF = "executive_brief"
    DB_LOCK_READINESS = "db_lock_readiness"
    ISSUE_ESCALATION = "issue_escalation"
    DAILY_DIGEST = "daily_digest"
    REGIONAL_SUMMARY = "regional_summary"
    CODING_STATUS = "coding_status"
    ENROLLMENT_TRACKER = "enrollment_tracker"


class OutputFormat(Enum):
    """Supported output formats"""
    HTML = "html"
    PDF = "pdf"
    WORD = "docx"
    POWERPOINT = "pptx"
    MARKDOWN = "md"
    JSON = "json"


@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    generated_by: str
    version: str = "1.0"
    classification: str = "Internal"
    expires_at: Optional[datetime] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'title': self.title,
            'generated_at': self.generated_at.isoformat(),
            'generated_by': self.generated_by,
            'version': self.version,
            'classification': self.classification,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'checksum': self.checksum
        }


@dataclass
class ReportTemplate:
    """Template definition"""
    template_id: str
    report_type: ReportType
    name: str
    description: str
    template_file: str
    required_variables: List[str]
    optional_variables: List[str] = field(default_factory=list)
    supported_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.HTML])
    sections: List[str] = field(default_factory=list)
    approver_role: Optional[str] = None
    generation_time_seconds: int = 30


@dataclass
class GeneratedReport:
    """Generated report output"""
    report_id: str
    metadata: ReportMetadata
    content: str
    format: OutputFormat
    file_path: Optional[str] = None
    generation_time_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'metadata': self.metadata.to_dict(),
            'format': self.format.value,
            'file_path': self.file_path,
            'generation_time_ms': self.generation_time_ms,
            'warnings': self.warnings,
            'content_length': len(self.content)
        }


class StringLoader(BaseLoader):
    """Load templates from strings (for inline templates)"""
    
    def __init__(self, templates: Dict[str, str]):
        self.templates = templates
    
    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)


class TemplateEngine:
    """Core engine for rendering reports"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml'])
        )
            
        self._register_filters(self.env)
        self.templates = self._initialize_templates()
        
    def _register_filters(self, env: Environment):
        """Register custom Jinja2 filters for formatting"""
        env.filters['format_date'] = lambda d: d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d
        env.filters['format_datetime'] = lambda d: d.strftime('%Y-%m-%d %H:%M:%S') if isinstance(d, datetime) else d
        env.filters['format_currency'] = lambda n: f"${n:,.2f}" if isinstance(n, (int, float)) else n
        env.filters['format_number'] = lambda n: f"{n:,}" if isinstance(n, (int, float)) else n
        env.filters['format_percent'] = lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x
        env.filters['format_decimal'] = lambda x, p=2: f"{x:.{p}f}" if isinstance(x, (int, float)) else x
        env.filters['priority_badge'] = self._priority_badge
        env.filters['risk_color'] = self._risk_to_color
        env.filters['dqi_band'] = self._dqi_to_band
        env.filters['title_case'] = lambda s: str(s).replace('_', ' ').title()
        env.filters['trend_arrow'] = lambda t: "↑" if t > 0 else "↓" if t < 0 else "→"
        
    def _priority_badge(self, priority: str) -> str:
        """Generate priority badge HTML"""
        p = str(priority).lower()
        colors = {
            'critical': 'background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;',
            'high': 'background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa;',
            'medium': 'background: #fef9c3; color: #854d0e; border: 1px solid #fef08a;',
            'low': 'background: #dcfce7; color: #166534; border: 1px solid #bbf7d0;',
        }
        style = colors.get(p, 'background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0;')
        return f'<span style="padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; {style}">{priority}</span>'

    def _risk_to_color(self, risk: str) -> str:
        colors = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#f97316', 'critical': '#ef4444'}
        return colors.get(str(risk).lower(), '#64748b')

    def _dqi_to_band(self, score: float) -> str:
        if score >= 90: return "Pristine"
        if score >= 80: return "Excellent"
        if score >= 70: return "Good"
        return "Critical"

    def _initialize_templates(self) -> Dict[str, ReportTemplate]:
        templates = [
            ReportTemplate("cra_monitoring", ReportType.CRA_MONITORING, "CRA Monitoring Report", "Site visit summary", "cra_monitoring.html", ['site_id', 'visit_date', 'cra_name', 'site_data']),
            ReportTemplate("site_performance", ReportType.SITE_PERFORMANCE, "Site Performance Summary", "Comprehensive metrics", "site_performance.html", ['site_id', 'period_start', 'period_end', 'metrics']),
            ReportTemplate("executive_brief", ReportType.EXECUTIVE_BRIEF, "Executive Brief", "High-level overview", "executive_brief.html", ['study_id', 'report_date', 'key_metrics']),
            ReportTemplate("db_lock_readiness", ReportType.DB_LOCK_READINESS, "DB Lock Readiness", "Assessment", "db_lock_readiness.html", ['study_id', 'target_date', 'readiness_data']),
            ReportTemplate("query_summary", ReportType.QUERY_SUMMARY, "Query Summary", "Status report", "query_summary.html", ['entity_id', 'query_data']),
            ReportTemplate("sponsor_update", ReportType.SPONSOR_UPDATE, "Sponsor Update", "Monthly pack", "sponsor_update.html", ['study_id', 'report_date', 'study_metrics']),
            ReportTemplate("meeting_pack", ReportType.MEETING_PACK, "Meeting Pack", "Slide deck", "meeting_pack.html", ['meeting_type', 'meeting_date', 'attendees', 'study_data']),
            ReportTemplate("safety_narrative", ReportType.SAFETY_NARRATIVE, "Safety Narrative", "Regulatory SAE", "safety_narrative.html", ['sae_id', 'patient_id', 'event_details']),
            ReportTemplate("regional_summary", ReportType.REGIONAL_SUMMARY, "Regional Summary", "Performance overview", "regional_summary.html", ['regions', 'recommendations']),
            ReportTemplate("coding_status", ReportType.CODING_STATUS, "Coding Status", "MedDRA status", "coding_status.html", ['coding_data']),
            ReportTemplate("enrollment_tracker", ReportType.ENROLLMENT_TRACKER, "Enrollment Tracker", "Target tracking", "enrollment_tracker.html", ['enrollment_data'])
        ]
        return {t.template_id: t for t in templates}

    def render(self, template_id: str, variables: Dict[str, Any], output_format: OutputFormat = OutputFormat.HTML, generated_by: str = "System") -> GeneratedReport:
        template_info = self.templates.get(template_id)
        if not template_info:
            raise ValueError(f"Template {template_id} not found")
            
        # Create metadata
        metadata = ReportMetadata(
            report_id=f"RPT-{template_id.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_type=template_info.report_type,
            title=template_info.name,
            generated_at=datetime.now(),
            generated_by=generated_by
        )
        
        render_vars = {**variables, '_metadata': metadata.to_dict(), 'now': datetime.now}
        template_str = self._get_inline_template(template_id)
        
        try:
            jinja_template = self.env.from_string(template_str)
            content = jinja_template.render(**render_vars)
            
            return GeneratedReport(
                report_id=metadata.report_id,
                metadata=metadata,
                content=content,
                format=output_format,
                generation_time_ms=0
            )
        except Exception as e:
            logger.error(f"Render failed: {e}")
            raise

    def _get_inline_template(self, template_id: str) -> str:
        common_css = """
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 40px; background: #f1f5f9; color: #334155; line-height: 1.5; }
        .container { max-width: 1000px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
        .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: #ffffff; padding: 40px; border-bottom: 6px solid #3b82f6; }
        .header h1 { margin: 0; font-size: 28px; text-transform: uppercase; letter-spacing: 1px; font-weight: 800; color: #ffffff; }
        .header .subtitle { margin-top: 10px; font-size: 15px; opacity: 0.9; font-weight: 500; color: #e2e8f0; }
        .content { padding: 40px; }
        .section { margin-bottom: 40px; }
        .section h2 { color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-top: 0; font-size: 18px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #ffffff; border: 1px solid #e2e8f0; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .card-value { font-size: 34px; font-weight: 800; color: #1e293b; line-height: 1; margin-bottom: 8px; }
        .card-label { font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }
        th { background: #34495e; padding: 15px 12px; text-align: left; font-size: 11px; text-transform: uppercase; color: #ffffff !important; font-weight: 700; border-bottom: 2px solid #e2e8f0; }
        td { padding: 15px 12px; border-bottom: 1px solid #f1f5f9; font-size: 14px; color: #334155; }
        .badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; display: inline-block; }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef9c3; color: #854d0e; }
        .badge-danger { background: #fee2e2; color: #991b1b; }
        .footer { background: #f8fafc; padding: 30px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }
        """

        templates = {
            "cra_monitoring": f"<html><style>{common_css}</style><body><div class='container'><div class='header'><h1>CRA Monitoring Report</h1><div class='subtitle'>Site: {{{{ site_id }}}} | Visit Date: {{{{ visit_date | format_date }}}}</div></div><div class='content'><div class='kpi-grid'><div class='card'><div class='card-value'>{{{{ site_data.total_patients }}}}</div><div class='card-label'>Total Patients</div></div><div class='card'><div class='card-value'>{{{{ site_data.dqi_score | format_decimal(1) }}}}</div><div class='card-label'>DQI Score</div></div><div class='card'><div class='card-value'>{{{{ site_data.clean_rate | format_percent }}}}</div><div class='card-label'>Clean Rate</div></div><div class='card'><div class='card-value'>{{{{ site_data.open_queries }}}}</div><div class='card-label'>Open Queries</div></div></div><div class='section'><h2>Data Quality Metrics</h2><table><thead><tr><th>Metric</th><th>Current</th><th>Target</th><th>Status</th></tr></thead><tbody>{{% for m in site_data.metrics %}}<tr><td>{{{{ m.name }}}}</td><td>{{{{ m.value | format_decimal(1) }}}}</td><td>{{{{ m.target | format_decimal(1) }}}}</td><td>{{% if m.value >= m.target %}}<span class='badge badge-success'>On Target</span>{{% else %}}<span class='badge badge-danger'>Below Target</span>{{% endif %}}</td></tr>{{% endfor %}}</tbody></table></div></div><div class='footer'>Report ID: {{{{ _metadata.report_id }}}}</div></div></body></html>",
            "site_performance": f"<html><style>{common_css}</style><body><div class='container'><div class='header'><h1>Site Performance Summary</h1><div class='subtitle'>Site: {{{{ site_id }}}} | Period: {{{{ period_start | format_date }}}} to {{{{ period_end | format_date }}}}</div></div><div class='content'><div class='kpi-grid'><div class='card'><div class='card-value'>{{{{ metrics.dqi | format_decimal(1) }}}}</div><div class='card-label'>DQI Score</div></div><div class='card'><div class='card-value'>{{{{ metrics.clean_rate | format_percent }}}}</div><div class='card-label'>Clean Rate</div></div><div class='card'><div class='card-value'>{{{{ metrics.query_resolution_days }}}}</div><div class='card-label'>Avg Query Days</div></div><div class='card'><div class='card-value'>{{{{ metrics.sdv_complete | format_percent }}}}</div><div class='card-label'>SDV Complete</div></div><div class='card'><div class='card-value'>{{{{ metrics.patients }}}}</div><div class='card-label'>Active Patients</div></div></div></div><div class='footer'>Report ID: {{{{ _metadata.report_id }}}}</div></div></body></html>",
            "executive_brief": f"<html><style>{common_css}</style><body><div class='container'><div class='header' style='background: #1e3a8a;'><h1>Executive Brief</h1><div class='subtitle'>Study: {{{{ study_id }}}} | Date: {{{{ report_date | format_date }}}}</div></div><div class='content'><div class='kpi-grid'><div class='card'><div class='card-value'>{{{{ key_metrics.patients | format_number }}}}</div><div class='card-label'>Total Enrolled</div></div><div class='card'><div class='card-value'>{{{{ key_metrics.dqi | format_decimal(1) }}}}</div><div class='card-label'>DQI Score</div></div><div class='card'><div class='card-value'>{{{{ key_metrics.clean_rate | format_percent }}}}</div><div class='card-label'>Clean Rate</div></div><div class='card'><div class='card-value'>{{{{ key_metrics.dblock_ready | format_percent }}}}</div><div class='card-label'>DB Lock Ready</div></div></div></div><div class='footer'>Report ID: {{{{ _metadata.report_id }}}}</div></div></body></html>",
            "db_lock_readiness": f"<html><style>{common_css}</style><body><div class='container'><div class='header' style='background: #0f172a;'><h1>DB Lock Readiness</h1><div class='subtitle'>Study: {{{{ study_id }}}} | Target: {{{{ target_date | format_date }}}}</div></div><div class='content'><div style='display: grid; grid-template-columns: 250px 1fr; gap: 40px;'><div class='card' style='background: #3b82f6; color: white;'><div class='card-value' style='color: white;'>{{{{ readiness_data.ready_rate | format_percent }}}}</div><div class='card-label' style='color: #dbeafe;'>Ready for Lock</div></div><div><h3>Readiness Breakdown</h3>{{% for cat in readiness_data.categories %}}<div style='margin-bottom: 10px;'><strong>{{{{ cat.name }}}}:</strong> {{{{ cat.rate | format_percent }}}}</div>{{% endfor %}}</div></div><div class='section'><h2>Site Summary</h2><table><thead><tr><th>Site</th><th>Patients</th><th>Ready</th><th>Status</th></tr></thead><tbody>{{% for s in readiness_data.sites %}}<tr><td>{{{{ s.site_id }}}}</td><td>{{{{ s.patients }}}}</td><td>{{{{ s.ready }}}}</td><td><span class='badge badge-success'>Verified</span></td></tr>{{% endfor %}}</tbody></table></div></div><div class='footer'>Report ID: {{{{ _metadata.report_id }}}}</div></div></body></html>",
            "query_summary": f"<html><style>{common_css}</style><body><div class='container'><div class='header' style='background: #ca8a04;'><h1>Query Summary</h1><div class='subtitle'>{{{{ entity_id }}}}</div></div><div class='content'><div class='kpi-grid'><div class='card'><div class='card-value'>{{{{ query_data.total }}}}</div><div class='card-label'>Total Lifetime</div></div><div class='card'><div class='card-value' style='color: #dc2626;'>{{{{ query_data.open }}}}</div><div class='card-label'>Open</div></div><div class='card'><div class='card-value' style='color: #059669;'>{{{{ query_data.resolved }}}}</div><div class='card-label'>Resolved</div></div><div class='card'><div class='card-value'>{{{{ query_data.avg_days }}}}</div><div class='card-label'>Avg Days Open</div></div></div><div class='section'><h2>Site Breakdown</h2><table><thead><tr><th>Site ID</th><th>Query Count</th><th>Status</th></tr></thead><tbody>{{% for q in query_list %}}<tr><td>{{{{ q.site_id }}}}</td><td>{{{{ q.query_count }}}}</td><td><span class='badge badge-warning'>Active</span></td></tr>{{% endfor %}}</tbody></table></div></div><div class='footer'>Report ID: {{{{ _metadata.report_id }}}}</div></div></body></html>",
            "sponsor_update": f"<html><style>{common_css}</style><body><div class='container'><div class='header'><h1>Sponsor Status Update</h1><div class='subtitle'>{{{{ study_id }}}} | {{{{ report_date | format_date }}}}</div></div><div class='content'><div class='kpi-grid'><div class='card'><div class='card-value'>{{{{ study_metrics.patients | format_number }}}}</div><div class='card-label'>Enrolled</div></div><div class='card'><div class='card-value'>{{{{ study_metrics.sites }}}}</div><div class='card-label'>Sites</div></div><div class='card'><div class='card-value'>{{{{ study_metrics.dqi | format_decimal(1) }}}}</div><div class='card-label'>DQI</div></div><div class='card'><div class='card-value'>{{{{ study_metrics.dblock_ready | format_percent }}}}</div><div class='card-label'>Readiness</div></div></div></div><div class='footer'>Report ID: {{{{ _metadata.report_id }}}}</div></div></body></html>",
            "meeting_pack": f"<html><style>{common_css} .slide {{ page-break-after: always; min-height: 800px; padding: 60px; }}</style><body><div class='container'><div class='slide' style='background: #1e293b; color: white; text-align: center;'><h1>{{{{ meeting_type }}}}</h1><p>{{{{ meeting_date | format_date }}}}</p><p style='margin-top: 40px;'>Attendees: {{{{ attendees | join(', ') }}}}</p></div><div class='slide'><h2>Agenda</h2>{{% for e in agenda %}}<div style='padding: 20px; background: #f8fafc; border: 1px solid #e2e8f0; margin-bottom: 10px;'>{{{{ loop.index }}}}. {{{{ e.item }}}} ({{{{ e.duration }}}})</div>{{% endfor %}}</div><div class='slide'><h2>Study Status</h2><div class='kpi-grid'><div class='card'><div class='card-value'>{{{{ study_data.patients | format_number }}}}</div><div class='card-label'>Patients</div></div><div class='card'><div class='card-value'>{{{{ study_data.dqi | format_decimal(1) }}}}</div><div class='card-label'>DQI</div></div><div class='card'><div class='card-value'>{{{{ study_data.clean_rate | format_percent }}}}</div><div class='card-label'>Clean Rate</div></div></div></div></div></body></html>",
            "safety_narrative": f"<html><style>{common_css} body {{ font-family: 'Georgia', serif; }} .box {{ padding: 30px; border: 1px solid #ddd; background: #fafafa; font-style: italic; }}</style><body><div class='container'><div class='header' style='background: #fff; color: #000; border-bottom: 2px solid #000;'><h1>SAE Narrative</h1><p>SAE ID: {{{{ sae_id }}}} | Patient: {{{{ patient_id }}}}</p></div><div class='content'><h3>Subject Info</h3><p>Age: {{{{ event_details.age | default('65') }}}} | Sex: {{{{ event_details.sex | default('M') }}}}</p><h3>Clinical Narrative</h3><div class='box'>{{% for p in narrative_summary %}}<p>{{{{ p }}}}</p>{{% endfor %}}</div></div></div></body></html>",
            "regional_summary": f"<html><style>{common_css}</style><body><div class='container'><div class='header' style='background: #3498db;'><h1>Regional Performance Summary</h1></div><div class='content'><table><thead><tr><th>Region</th><th>DQI</th><th>Sites</th><th>Patients</th></tr></thead><tbody>{{% for r in regions %}}<tr><td>{{{{ r.region }}}}</td><td>{{{{ r.dqi | format_decimal(1) }}}}</td><td>{{{{ r.sites }}}}</td><td>{{{{ r.patients | format_number }}}}</td></tr>{{% endfor %}}</tbody></table></div></div></body></html>",
            "coding_status": f"<html><style>{common_css}</style><body><div class='container'><div class='header' style='background: #9b59b6;'><h1>Coding Status</h1></div><div class='content'><h3>MedDRA Completion: {{{{ coding_data.meddra.completion | format_decimal(1) }}}}%</h3><h3>WHODRA Completion: {{{{ coding_data.whodra.completion | format_decimal(1) }}}}%</h3></div></div></body></html>",
            "enrollment_tracker": f"<html><style>{common_css}</style><body><div class='container'><div class='header' style='background: #27ae60;'><h1>Enrollment Tracker</h1></div><div class='content'><h3>Total Enrolled: {{{{ enrollment_data.total_enrolled }}}} / {{{{ enrollment_data.target_enrolled }}}}</h3><p>Current Pace: {{{{ enrollment_data.current_pace }}}} patients/month</p></div></div></body></html>"
        }
        return templates.get(template_id, "<html><body>Default Template</body></html>")

def get_template_engine() -> TemplateEngine:
    return TemplateEngine()
