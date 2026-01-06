"""
Tool Registry for TRIALPULSE NEXUS 10X
Phase 5.2: LangGraph Agent Framework - FIXED

Defines the tools that agents can use to interact with data and systems.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    category: str = "general"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        try:
            result = self.function(**kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(success=False, error=str(e))


class ToolRegistry:
    """
    Registry of all available tools for agents.
    
    Tools are organized by category:
    - data: Data retrieval and analysis
    - search: Vector search and RAG
    - analytics: DQI, cascade, benchmarks
    - ml: ML model predictions
    - action: Executable actions
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the tool registry"""
        self.data_dir = data_dir or PROJECT_ROOT / "data" / "processed"
        self.tools: Dict[str, Tool] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Register all tools
        self._register_data_tools()
        self._register_search_tools()
        self._register_analytics_tools()
        self._register_ml_tools()
        self._register_action_tools()
        
        logger.info(f"ToolRegistry initialized with {len(self.tools)} tools")
    
    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        return tool.execute(**kwargs)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List all tools, optionally filtered by category"""
        tools = []
        for name, tool in self.tools.items():
            if category is None or tool.category == category:
                tools.append({
                    "name": name,
                    "description": tool.description,
                    "category": tool.category,
                    "requires_approval": tool.requires_approval
                })
        return tools
    
    def get_tools_for_llm(self) -> str:
        """Get tool descriptions formatted for LLM context"""
        lines = ["Available Tools:"]
        for name, tool in sorted(self.tools.items(), key=lambda x: x[1].category):
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)
    
    # === Helper to convert numpy types to Python native ===
    
    def _to_native(self, obj):
        """Convert numpy types to Python native types for serialization"""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # === Data Loading Helper ===
    
    def _load_data(self, filename: str, subdir: str = "") -> Optional[pd.DataFrame]:
        """Load data file with caching. Returns None if file doesn't exist."""
        cache_key = f"{subdir}/{filename}"
        
        if cache_key not in self._data_cache:
            if subdir:
                filepath = self.data_dir / subdir / filename
            else:
                filepath = self.data_dir / filename
            
            if filepath.exists():
                try:
                    self._data_cache[cache_key] = pd.read_parquet(filepath)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
                    return None
            else:
                logger.debug(f"Data file not found: {filepath}")
                return None
        
        return self._data_cache[cache_key]
    
    def _get_primary_data(self) -> Optional[pd.DataFrame]:
        """Get the primary patient data file with all columns"""
        # Try multiple possible data files in order of preference
        files_to_try = [
            ("patient_dqi_enhanced.parquet", "analytics"),
            ("patient_clean_status.parquet", "analytics"),
            ("patient_dblock_status.parquet", "analytics"),
            ("unified_patient_record.parquet", "upr"),
        ]
        
        for filename, subdir in files_to_try:
            df = self._load_data(filename, subdir)
            if df is not None:
                return df
        
        logger.warning("No primary data file available")
        return None
    
    # === Data Tools ===
    
    def _register_data_tools(self):
        """Register data retrieval tools"""
        
        def get_patient(patient_key: str) -> Dict[str, Any]:
            """Get patient data by patient_key"""
            df = self._get_primary_data()
            if df is None:
                return {"error": "Data not available. Please run pipelines first."}
            
            patient = df[df["patient_key"] == patient_key]
            if len(patient) == 0:
                return {"error": f"Patient {patient_key} not found"}
            
            row = patient.iloc[0]
            return {
                "patient_key": patient_key,
                "study_id": row.get("study_id"),
                "site_id": row.get("site_id"),
                "subject_status": row.get("subject_status_clean"),
                "dqi_score": float(row.get("dqi_score", 0)),
                "dqi_band": row.get("dqi_band"),
                "risk_level": row.get("risk_level"),
                "tier1_clinical_clean": bool(row.get("tier1_clinical_clean", False)),
                "tier2_operational_clean": bool(row.get("tier2_operational_clean", False)),
                "total_queries": float(row.get("total_queries", 0)),
                "total_issues": int(row.get("total_issues_all_sources", 0)),
                "dqi_primary_issue": row.get("dqi_primary_issue"),
                "tier1_blocking_reason": row.get("tier1_blocking_reason"),
                "tier2_blocking_reason": row.get("tier2_blocking_reason"),
                "db_lock_status": row.get("db_lock_status"),
            }
        
        self.register(Tool(
            name="get_patient",
            description="Get detailed patient data including DQI, clean status, and issues",
            function=get_patient,
            parameters={"patient_key": "str - Patient identifier (Study_X|Site_XXX|Subject_XXXX)"},
            category="data"
        ))
        
        def get_site_summary(site_id: str) -> Dict[str, Any]:
            """Get site summary statistics"""
            df = self._get_primary_data()
            if df is None:
                return {"error": "Data not available. Please run pipelines first."}
            
            site_data = df[df["site_id"] == site_id]
            
            if len(site_data) == 0:
                return {"error": f"Site {site_id} not found"}
            
            return {
                "site_id": site_id,
                "study_id": site_data["study_id"].iloc[0],
                "patient_count": len(site_data),
                "avg_dqi": float(site_data["dqi_score"].mean()),
                "tier1_clean_rate": float(site_data["tier1_clinical_clean"].mean() * 100) if "tier1_clinical_clean" in site_data else 0,
                "tier2_clean_rate": float(site_data["tier2_operational_clean"].mean() * 100) if "tier2_operational_clean" in site_data else 0,
                "total_queries": int(site_data["total_queries"].sum()),
                "patients_with_issues": int((site_data["total_issues_all_sources"] > 0).sum()),
                "risk_high_critical": int(site_data["risk_level"].isin(["High", "Critical"]).sum()),
                "db_lock_ready": int((site_data["db_lock_status"] == "Ready").sum()) if "db_lock_status" in site_data else 0,
            }
        
        self.register(Tool(
            name="get_site_summary",
            description="Get summary statistics for a site including patient count, DQI, and issues",
            function=get_site_summary,
            parameters={"site_id": "str - Site identifier (e.g., Site_101)"},
            category="data"
        ))
        
        def get_study_summary(study_id: str) -> Dict[str, Any]:
            """Get study summary statistics"""
            df = self._get_primary_data()
            if df is None:
                return {"error": "Data not available. Please run pipelines first."}
            
            study_data = df[df["study_id"] == study_id]
            
            if len(study_data) == 0:
                return {"error": f"Study {study_id} not found"}
            
            return self._to_native({
                "study_id": study_id,
                "patient_count": len(study_data),
                "site_count": int(study_data["site_id"].nunique()),
                "avg_dqi": float(study_data["dqi_score"].mean()),
                "dqi_std": float(study_data["dqi_score"].std()),
                "dqi_min": float(study_data["dqi_score"].min()),
                "dqi_max": float(study_data["dqi_score"].max()),
                "tier1_clean_rate": float(study_data["tier1_clinical_clean"].mean() * 100) if "tier1_clinical_clean" in study_data else 0,
                "tier2_clean_rate": float(study_data["tier2_operational_clean"].mean() * 100) if "tier2_operational_clean" in study_data else 0,
                "total_queries": int(study_data["total_queries"].sum()),
                "patients_with_issues": int((study_data["total_issues_all_sources"] > 0).sum()),
                "issue_rate": float((study_data["total_issues_all_sources"] > 0).mean() * 100),
                "risk_distribution": study_data["risk_level"].value_counts().to_dict(),
                "dqi_band_distribution": study_data["dqi_band"].value_counts().to_dict(),
                "top_primary_issues": study_data["dqi_primary_issue"].value_counts().head(5).to_dict(),
            })
        
        self.register(Tool(
            name="get_study_summary",
            description="Get summary statistics for a study including site count and issue rates",
            function=get_study_summary,
            parameters={"study_id": "str - Study identifier (e.g., Study_21)"},
            category="data"
        ))
        
        def get_high_priority_patients(limit: int = 20) -> List[Dict[str, Any]]:
            """Get patients with high/critical risk or low DQI scores"""
            df = self._get_primary_data()
            if df is None:
                return [{"error": "Data not available. Please run pipelines first."}]
            
            # Filter for high priority patients
            high_priority = df[
                (df["risk_level"].isin(["High", "Critical"])) | 
                (df["dqi_score"] < 70)
            ].copy()
            
            # Sort by DQI score (lowest first)
            high_priority = high_priority.sort_values("dqi_score", ascending=True)
            
            results = []
            for _, row in high_priority.head(limit).iterrows():
                results.append({
                    "patient_key": row["patient_key"],
                    "study_id": row["study_id"],
                    "site_id": row["site_id"],
                    "risk_level": row["risk_level"],
                    "dqi_score": float(row["dqi_score"]),
                    "dqi_band": row["dqi_band"],
                    "total_issues": int(row.get("total_issues_all_sources", 0)),
                    "total_queries": int(row.get("total_queries", 0)),
                    "dqi_primary_issue": row.get("dqi_primary_issue"),
                })
            
            return results
        
        self.register(Tool(
            name="get_high_priority_patients",
            description="Get list of patients with high risk or high cascade impact",
            function=get_high_priority_patients,
            parameters={"limit": "int - Maximum number of patients to return (default: 20)"},
            category="data"
        ))
        
        def get_overall_summary() -> Dict[str, Any]:
            """Get overall summary across all studies"""
            df = self._get_primary_data()
            if df is None:
                return {"error": "Data not available. Please run pipelines first."}
            
            return self._to_native({
                "total_patients": len(df),
                "total_studies": int(df["study_id"].nunique()),
                "total_sites": int(df["site_id"].nunique()),
                "avg_dqi": float(df["dqi_score"].mean()),
                "dqi_std": float(df["dqi_score"].std()),
                "tier1_clean_rate": float(df["tier1_clinical_clean"].mean() * 100) if "tier1_clinical_clean" in df else 0,
                "tier2_clean_rate": float(df["tier2_operational_clean"].mean() * 100) if "tier2_operational_clean" in df else 0,
                "patients_with_issues": int((df["total_issues_all_sources"] > 0).sum()),
                "patients_clean": int((df["total_issues_all_sources"] == 0).sum()),
                "risk_distribution": df["risk_level"].value_counts().to_dict(),
                "dqi_band_distribution": df["dqi_band"].value_counts().to_dict(),
                "top_issues": df["dqi_primary_issue"].value_counts().head(5).to_dict(),
            })
        
        self.register(Tool(
            name="get_overall_summary",
            description="Get overall summary statistics across all studies",
            function=get_overall_summary,
            parameters={},
            category="data"
        ))
        
        def diagnose_study_dqi(study_id: str) -> Dict[str, Any]:
            """
            Comprehensive DQI diagnostic for a study.
            Returns detailed analysis comparing to portfolio average, 
            identifying anomalies, and ranking hypotheses.
            """
            df = self._get_primary_data()
            if df is None:
                return {"error": "Data not available. Please run pipelines first."}
            
            # Get study data
            study_data = df[df["study_id"] == study_id]
            if len(study_data) == 0:
                return {"error": f"Study {study_id} not found"}
            
            # Calculate study metrics
            study_dqi = study_data["dqi_score"].mean()
            portfolio_dqi = df["dqi_score"].mean()
            dqi_diff = study_dqi - portfolio_dqi
            
            # Component scores
            component_cols = [
                "dqi_safety_raw", "dqi_query_raw", "dqi_completeness_raw",
                "dqi_coding_raw", "dqi_lab_raw", "dqi_sdv_raw", 
                "dqi_signature_raw", "dqi_edrr_raw"
            ]
            
            component_analysis = {}
            for col in component_cols:
                if col in study_data.columns:
                    study_avg = study_data[col].mean()
                    portfolio_avg = df[col].mean()
                    component_name = col.replace("dqi_", "").replace("_raw", "")
                    component_analysis[component_name] = {
                        "study_avg": round(float(study_avg), 1),
                        "portfolio_avg": round(float(portfolio_avg), 1),
                        "difference": round(float(study_avg - portfolio_avg), 1),
                        "is_below_average": bool(study_avg < portfolio_avg)
                    }
            
            # Find worst components
            worst_components = sorted(
                component_analysis.items(),
                key=lambda x: x[1]["difference"]
            )[:3]
            
            # Issue breakdown
            issue_counts = study_data["dqi_primary_issue"].value_counts().to_dict()
            
            # Site performance
            site_dqi = study_data.groupby("site_id")["dqi_score"].mean().sort_values()
            worst_sites = site_dqi.head(5).to_dict()
            best_sites = site_dqi.tail(5).to_dict()
            
            # Risk distribution
            risk_dist = study_data["risk_level"].value_counts().to_dict()
            
            # Query analysis
            total_queries = study_data["total_queries"].sum()
            dm_queries = study_data["dm_queries"].sum() if "dm_queries" in study_data else 0
            clinical_queries = study_data["clinical_queries"].sum() if "clinical_queries" in study_data else 0
            
            # Generate hypotheses based on data
            hypotheses = []
            
            # Check for query issues
            if component_analysis.get("query", {}).get("is_below_average"):
                hypotheses.append({
                    "hypothesis": "High query volume impacting DQI",
                    "confidence": 85 if total_queries > 1000 else 60,
                    "evidence": f"Total queries: {int(total_queries)}, Study avg query score: {component_analysis['query']['study_avg']}"
                })
            
            # Check for SDV issues
            if component_analysis.get("sdv", {}).get("is_below_average"):
                sdv_score = component_analysis["sdv"]["study_avg"]
                hypotheses.append({
                    "hypothesis": "SDV backlog causing DQI drop",
                    "confidence": 90 if sdv_score < 80 else 65,
                    "evidence": f"SDV score: {sdv_score}, vs portfolio {component_analysis['sdv']['portfolio_avg']}"
                })
            
            # Check for signature issues
            if component_analysis.get("signature", {}).get("is_below_average"):
                sig_score = component_analysis["signature"]["study_avg"]
                hypotheses.append({
                    "hypothesis": "Signature delays impacting quality",
                    "confidence": 75 if sig_score < 85 else 55,
                    "evidence": f"Signature score: {sig_score}"
                })
            
            # Check for coding issues
            if component_analysis.get("coding", {}).get("is_below_average"):
                hypotheses.append({
                    "hypothesis": "Medical coding backlog",
                    "confidence": 70,
                    "evidence": f"Coding score: {component_analysis['coding']['study_avg']}"
                })
            
            # Sort hypotheses by confidence
            hypotheses = sorted(hypotheses, key=lambda x: x["confidence"], reverse=True)
            
            # Convert all numpy types to native Python for serialization
            return self._to_native({
                "study_id": study_id,
                "data_gathered": {
                    "patient_count": len(study_data),
                    "site_count": study_data["site_id"].nunique(),
                    "dqi_score": round(float(study_dqi), 2),
                    "dqi_std": round(float(study_data["dqi_score"].std()), 2),
                    "total_queries": int(total_queries),
                    "dm_queries": int(dm_queries),
                    "clinical_queries": int(clinical_queries),
                },
                "comparison": {
                    "portfolio_dqi_avg": round(float(portfolio_dqi), 2),
                    "difference_from_avg": round(float(dqi_diff), 2),
                    "percentile_rank": round(float((df.groupby("study_id")["dqi_score"].mean() <= study_dqi).mean() * 100), 1),
                    "status": "ABOVE AVERAGE" if dqi_diff > 0 else "BELOW AVERAGE"
                },
                "component_analysis": component_analysis,
                "anomalies_detected": [w[0] for w in worst_components if w[1]["is_below_average"]],
                "worst_components": {w[0]: w[1] for w in worst_components},
                "issue_breakdown": issue_counts,
                "site_analysis": {
                    "worst_performing": worst_sites,
                    "best_performing": best_sites
                },
                "risk_distribution": risk_dist,
                "hypotheses_ranked": hypotheses if hypotheses else [{"hypothesis": "Study performing at or above average", "confidence": 90, "evidence": f"DQI {round(float(study_dqi), 1)} vs portfolio {round(float(portfolio_dqi), 1)}"}]
            })
        
        self.register(Tool(
            name="diagnose_study_dqi",
            description="Comprehensive DQI diagnostic for a study - analyzes components, compares to portfolio, finds anomalies, and generates ranked hypotheses",
            function=diagnose_study_dqi,
            parameters={"study_id": "str - Study identifier (e.g., Study_21)"},
            category="analytics"
        ))
    
    # === Search Tools ===
    
    def _register_search_tools(self):
        """Register search and RAG tools"""
        
        def search_resolutions(issue_type: str, limit: int = 5) -> List[Dict[str, Any]]:
            """Search resolution templates by issue type"""
            try:
                genome_dir = self.data_dir / "analytics" / "resolution_genome"
                templates_file = genome_dir / "resolution_templates.json"
                
                if templates_file.exists():
                    with open(templates_file) as f:
                        templates = json.load(f)
                    
                    # Filter by issue type
                    matches = [t for t in templates if t.get("issue_type") == issue_type]
                    return matches[:limit] if matches else [{"message": f"No templates for {issue_type}"}]
                else:
                    return [{"error": "Resolution templates not found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="search_resolutions",
            description="Search for resolution templates by issue type",
            function=search_resolutions,
            parameters={
                "issue_type": "str - Issue type (e.g., sdv_incomplete, open_queries)",
                "limit": "int - Maximum results (default: 5)"
            },
            category="search"
        ))
        
        def search_patterns(pattern_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
            """Search for known patterns"""
            try:
                pattern_dir = self.data_dir / "analytics" / "pattern_library"
                patterns_file = pattern_dir / "pattern_definitions.csv"
                
                if patterns_file.exists():
                    df = pd.read_csv(patterns_file)
                    if pattern_type:
                        df = df[df["category"] == pattern_type]
                    return df.head(limit).to_dict("records")
                else:
                    return [{"error": "Pattern library not found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="search_patterns",
            description="Search for known issue patterns from pattern library",
            function=search_patterns,
            parameters={
                "pattern_type": "str - Pattern category (optional)",
                "limit": "int - Maximum results (default: 10)"
            },
            category="search"
        ))
        
        def search_knowledge(query: str, source: str = None, limit: int = 5) -> List[Dict[str, Any]]:
            """Search RAG knowledge base"""
            try:
                rag_dir = self.data_dir / "knowledge" / "rag"
                docs_file = rag_dir / "rag_documents.json"
                
                if docs_file.exists():
                    with open(docs_file) as f:
                        docs = json.load(f)
                    
                    # Simple keyword matching
                    query_lower = query.lower()
                    matches = []
                    for doc in docs:
                        text = doc.get("text", "").lower()
                        if query_lower in text:
                            score = text.count(query_lower) / max(len(text.split()), 1)
                            matches.append({
                                "text": doc.get("text", "")[:300],
                                "source": doc.get("metadata", {}).get("source", "unknown"),
                                "category": doc.get("metadata", {}).get("category", "unknown"),
                                "score": round(score, 4)
                            })
                    
                    matches.sort(key=lambda x: x["score"], reverse=True)
                    if source:
                        matches = [m for m in matches if m["source"] == source]
                    
                    return matches[:limit] if matches else [{"message": "No matches found"}]
                else:
                    return [{"error": "RAG knowledge base not found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="search_knowledge",
            description="Search regulatory guidelines, SOPs, and protocol knowledge",
            function=search_knowledge,
            parameters={
                "query": "str - Search query",
                "source": "str - Filter by source (ich_gcp, protocol, sop) (optional)",
                "limit": "int - Maximum results (default: 5)"
            },
            category="search"
        ))
    
    # === Analytics Tools ===
    
    def _register_analytics_tools(self):
        """Register analytics tools"""
        
        def get_cascade_impact(patient_key: str) -> Dict[str, Any]:
            """Get cascade impact analysis for a patient"""
            try:
                df = self._get_primary_data()
                patient = df[df["patient_key"] == patient_key]
                
                if len(patient) == 0:
                    return {"error": f"Patient {patient_key} not found"}
                
                row = patient.iloc[0]
                return {
                    "patient_key": patient_key,
                    "cascade_impact_score": float(row.get("cascade_impact_score", 0)),
                    "cascade_issue_count": int(row.get("cascade_issue_count", 0)),
                    "cascade_path_length": int(row.get("cascade_path_length", 0)),
                    "cascade_cluster": row.get("cascade_cluster"),
                    "cascade_critical_path": row.get("cascade_critical_path"),
                    "cascade_primary_blocker": row.get("cascade_primary_blocker"),
                    "cascade_recommendation": row.get("cascade_recommendation"),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_cascade_impact",
            description="Get cascade impact analysis showing how fixing one issue unlocks others",
            function=get_cascade_impact,
            parameters={"patient_key": "str - Patient identifier"},
            category="analytics"
        ))
        
        def get_site_benchmark(site_id: str) -> Dict[str, Any]:
            """Get site benchmark compared to peers"""
            try:
                df = self._load_data("site_benchmarks.parquet", "analytics")
                site = df[df["site_id"] == site_id]
                
                if len(site) == 0:
                    return {"error": f"Site {site_id} not found in benchmarks"}
                
                row = site.iloc[0]
                return {
                    "site_id": site_id,
                    "study_id": row.get("study_id"),
                    "patient_count": int(row.get("patient_count", 0)),
                    "composite_score": float(row.get("composite_score", 0)),
                    "composite_percentile": float(row.get("composite_percentile", 0)),
                    "performance_tier": row.get("performance_tier"),
                    "dqi_mean": float(row.get("dqi_mean", 0)),
                    "tier1_clean_rate": float(row.get("tier1_clean_rate", 0)),
                    "tier2_clean_rate": float(row.get("tier2_clean_rate", 0)),
                    "ready_rate": float(row.get("ready_rate", 0)),
                    "overall_rank": int(row.get("overall_rank", 0)),
                    "study_rank": int(row.get("study_rank", 0)),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_site_benchmark",
            description="Get site performance benchmark compared to peer sites",
            function=get_site_benchmark,
            parameters={"site_id": "str - Site identifier"},
            category="analytics"
        ))
        
        def get_dblock_projection(study_id: str = None) -> Dict[str, Any]:
            """Get DB Lock readiness projection"""
            try:
                df = self._get_primary_data()
                
                if study_id:
                    df = df[df["study_id"] == study_id]
                
                if len(df) == 0:
                    return {"error": "No data found for projection"}
                
                eligible = df[df["db_lock_eligible"] == True]
                
                return {
                    "study_id": study_id or "All Studies",
                    "total_patients": len(df),
                    "eligible_patients": len(eligible),
                    "ready_now": int((eligible["db_lock_status"] == "Ready").sum()),
                    "ready_rate": float((eligible["db_lock_status"] == "Ready").mean() * 100) if len(eligible) > 0 else 0,
                    "pending": int((eligible["db_lock_status"] == "Pending").sum()),
                    "blocked": int((eligible["db_lock_status"] == "Blocked").sum()),
                    "not_eligible": int((df["db_lock_eligible"] == False).sum()),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_dblock_projection",
            description="Get DB Lock readiness status and projection",
            function=get_dblock_projection,
            parameters={"study_id": "str - Study identifier (optional, defaults to all)"},
            category="analytics"
        ))
    
    # === ML Tools ===
    
    def _register_ml_tools(self):
        """Register ML prediction tools"""
        
        def predict_risk(patient_key: str) -> Dict[str, Any]:
            """Get risk prediction for a patient"""
            try:
                df = self._get_primary_data()
                patient = df[df["patient_key"] == patient_key]
                
                if len(patient) == 0:
                    return {"error": f"Patient {patient_key} not found"}
                
                row = patient.iloc[0]
                return {
                    "patient_key": patient_key,
                    "risk_level": row.get("risk_level", "Unknown"),
                    "dqi_score": float(row.get("dqi_score", 0)),
                    "dqi_band": row.get("dqi_band"),
                    "cascade_issue_count": int(row.get("cascade_issue_count", 0)),
                    "primary_issue": row.get("dqi_primary_issue"),
                    "tier1_clean": bool(row.get("tier1_clean", False)),
                    "tier2_clean": bool(row.get("tier2_clean", False)),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="predict_risk",
            description="Get risk assessment for a patient",
            function=predict_risk,
            parameters={"patient_key": "str - Patient identifier"},
            category="ml"
        ))
        
        def detect_anomalies(site_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
            """Detect anomalies at patient or site level"""
            try:
                df = self._load_data("patient_anomalies.parquet", "analytics")
                
                if site_id:
                    df = df[df["site_id"] == site_id]
                
                # Get anomalies (is_anomaly = True or high severity)
                anomalies = df[
                    (df["is_anomaly"] == True) | 
                    (df["anomaly_severity"].isin(["Critical", "High"]))
                ]
                
                # Sort by score
                anomalies = anomalies.sort_values("anomaly_score_ensemble", ascending=False)
                
                results = []
                for _, row in anomalies.head(limit).iterrows():
                    results.append({
                        "patient_key": row["patient_key"],
                        "site_id": row["site_id"],
                        "study_id": row["study_id"],
                        "anomaly_severity": row["anomaly_severity"],
                        "anomaly_score": float(row["anomaly_score_ensemble"]),
                        "is_anomaly": bool(row["is_anomaly"]),
                    })
                
                return results if results else [{"message": "No anomalies found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="detect_anomalies",
            description="Detect anomalous patterns at patient or site level",
            function=detect_anomalies,
            parameters={
                "site_id": "str - Site identifier (optional)",
                "limit": "int - Maximum results (default: 10)"
            },
            category="ml"
        ))
    
    # === Action Tools ===
    
    def _register_action_tools(self):
        """Register action execution tools"""
        
        def draft_query_email(site_id: str, issue_summary: str, recipient: str = "Site Coordinator") -> Dict[str, Any]:
            """Draft an email about pending queries"""
            return {
                "type": "email",
                "recipient": recipient,
                "subject": f"Action Required: Pending Queries at {site_id}",
                "body": f"""Dear {recipient},

This is a reminder regarding pending data quality items at {site_id}.

Summary:
{issue_summary}

Please review and resolve these items at your earliest convenience.

Best regards,
Clinical Trial Team""",
                "requires_approval": True,
                "status": "draft"
            }
        
        self.register(Tool(
            name="draft_query_email",
            description="Draft an email reminder for pending queries",
            function=draft_query_email,
            parameters={
                "site_id": "str - Site identifier",
                "issue_summary": "str - Summary of the issues",
                "recipient": "str - Recipient role (default: Site Coordinator)"
            },
            requires_approval=True,
            category="action"
        ))
        
        def create_task(title: str, description: str, assignee: str, priority: str = "medium", due_days: int = 7) -> Dict[str, Any]:
            """Create a task for follow-up"""
            from datetime import datetime, timedelta
            
            return {
                "task_id": f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "title": title,
                "description": description,
                "assignee": assignee,
                "priority": priority,
                "due_date": (datetime.now() + timedelta(days=due_days)).isoformat(),
                "status": "pending",
                "requires_approval": True
            }
        
        self.register(Tool(
            name="create_task",
            description="Create a follow-up task for an action item",
            function=create_task,
            parameters={
                "title": "str - Task title",
                "description": "str - Task description",
                "assignee": "str - Person or role to assign",
                "priority": "str - Priority (low/medium/high/critical)",
                "due_days": "int - Days until due (default: 7)"
            },
            requires_approval=True,
            category="action"
        ))
        
        def log_investigation(patient_key: str, finding: str, next_steps: str) -> Dict[str, Any]:
            """Log an investigation finding"""
            from datetime import datetime
            
            return {
                "investigation_id": f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patient_key": patient_key,
                "timestamp": datetime.now().isoformat(),
                "finding": finding,
                "next_steps": next_steps,
                "status": "logged"
            }
        
        self.register(Tool(
            name="log_investigation",
            description="Log an investigation finding for audit trail",
            function=log_investigation,
            parameters={
                "patient_key": "str - Patient identifier",
                "finding": "str - Investigation finding",
                "next_steps": "str - Recommended next steps"
            },
            category="action"
        ))


# Singleton instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry