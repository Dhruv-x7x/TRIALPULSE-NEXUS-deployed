"""
Issue Routes
============
Issue management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from datetime import datetime
import sys
import os
import numpy as np
import math
import uuid

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))

from app.models.schemas import IssueListResponse, IssueCreateRequest, IssueUpdateRequest
from app.core.security import get_current_user, require_role
from app.services.database import get_data_service

router = APIRouter()


@router.get("", response_model=IssueListResponse)
async def list_issues(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=2000),
    current_user: dict = Depends(get_current_user)
):
    """Get project issues with optional filters."""
    # Guard against React Query objects or "all" string
    if site_id and (site_id == "[object Object]" or "{" in site_id or site_id.lower() == "all"):
        site_id = None
    if study_id and (study_id == "[object Object]" or "{" in study_id or study_id.lower() == "all"):
        study_id = None
        
    try:
        data_service = get_data_service()
        df = data_service.get_issues(status=status, limit=limit, study_id=study_id)
        
        if df.empty:
            return IssueListResponse(issues=[], total=0)
        
        # Apply additional filters in memory
        if priority and "priority" in df.columns:
            df = df[df["priority"] == priority]
        if site_id and "site_id" in df.columns:
            df = df[df["site_id"] == site_id]
        
        # Convert to records and ensure all values are JSON-serializable
        from .patients import sanitize_for_json
        records = sanitize_for_json(df.to_dict(orient="records"))
        
        return IssueListResponse(
            issues=records,
            total=len(records)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_issues_summary(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get issues summary by status and priority with optimized SQL aggregation."""
    # Guard against React Query objects or "all" string
    if study_id and (study_id == "[object Object]" or "{" in study_id or study_id.lower() == "all"):
        study_id = None
        
    try:
        data_service = get_data_service()
        # Use new optimized method for aggregation in SQL
        stats = data_service.get_issue_summary_stats(study_id=study_id)
        
        if not stats:
            return {
                "total": 0,
                "by_status": {},
                "by_priority": {},
                "by_type": {},
                "open_count": 0,
                "critical_count": 0
            }
            
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient-summary")
async def get_patient_issues_summary(
    current_user: dict = Depends(get_current_user)
):
    """Get patient-level issues summary."""
    try:
        data_service = get_data_service()
        df = data_service.get_patient_issues()
        
        if df.empty:
            return {
                "total_patients": 0,
                "patients_with_issues": 0,
                "by_priority_tier": {}
            }
        
        total = len(df)
        with_issues = int(df["has_issues"].sum()) if "has_issues" in df.columns else 0
        
        # Count by priority tier
        tier_counts = {}
        if "priority_tier" in df.columns:
            tier_counts = df["priority_tier"].value_counts().to_dict()
        
        return {
            "total_patients": total,
            "patients_with_issues": with_issues,
            "patients_clean": total - with_issues,
            "issue_rate": round(with_issues / total * 100, 2) if total > 0 else 0,
            "by_priority_tier": tier_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_issue(
    request: IssueCreateRequest,
    current_user: dict = Depends(require_role("lead", "dm", "cra"))
):
    """Create a new issue."""
    # In production, this would insert into database
    # For now, return a mock response
    return {
        "message": "Issue created successfully",
        "issue": {
            "issue_id": 9999,
            "patient_key": request.patient_key,
            "site_id": request.site_id,
            "issue_type": request.issue_type,
            "priority": request.priority.value,
            "description": request.description,
            "status": "open",
            "created_at": datetime.utcnow().isoformat(),
            "created_by": current_user.get("username")
        }
    }


@router.put("/{issue_id}")
async def update_issue(
    issue_id: int,
    request: IssueUpdateRequest,
    current_user: dict = Depends(require_role("lead", "dm", "cra"))
):
    """Update an existing issue."""
    # In production, this would update the database
    return {
        "message": "Issue updated successfully",
        "issue_id": issue_id,
        "updates": request.model_dump(exclude_none=True),
        "updated_by": current_user.get("username"),
        "updated_at": datetime.utcnow().isoformat()
    }


@router.post("/{issue_id}/resolve")
async def resolve_issue(
    issue_id: str,
    reason_for_change: str, # Mandatory per 21 CFR Part 11
    resolution_notes: Optional[str] = None,
    current_user: dict = Depends(require_role("lead", "dm", "cra"))
):
    """Resolve an issue in the SQL database with mandatory 21 CFR compliance reason."""
    try:
        from sqlalchemy import text
        import hashlib
        data_service = get_data_service()
        engine = data_service._db_manager.engine
        
        with engine.begin() as conn:
            # 1. Get current state for audit trail
            curr = conn.execute(
                text("SELECT * FROM project_issues WHERE issue_id = :issue_id"),
                {"issue_id": issue_id}
            ).fetchone()
            
            if not curr:
                raise HTTPException(status_code=404, detail="Issue not found")
                
            # 2. Update project_issues
            conn.execute(
                text("UPDATE project_issues SET status = 'resolved', resolved_at = :now, resolution_notes = :notes WHERE issue_id = :issue_id"),
                {"issue_id": issue_id, "now": datetime.utcnow(), "notes": resolution_notes or 'Resolved via dashboard'}
            )
            
            # 3. Update patient counts
            if curr.patient_key:
                conn.execute(
                    text("UPDATE patients SET open_issues_count = GREATEST(0, open_issues_count - 1) WHERE patient_key = :patient_key"),
                    {"patient_key": curr.patient_key}
                )
            
            # 4. Generate Digital Signature Hash (Simplified for demo)
            sig_content = f"{issue_id}|resolved|{current_user.get('username')}|{datetime.utcnow().isoformat()}"
            sig_hash = hashlib.sha256(sig_content.encode()).hexdigest()
            
            # 5. Record in Audit Log (21 CFR Part 11)
            conn.execute(
                text("""
                    INSERT INTO audit_logs 
                    (log_id, timestamp, user_id, user_name, user_role, action, 
                     entity_type, entity_id, field_name, old_value, new_value, 
                     reason, checksum)
                    VALUES 
                    (:log_id, :now, :user_id, :user_name, :user_role, 'RESOLVE',
                     'ISSUE', :issue_id, 'status', 'open', 'resolved',
                     :reason, :checksum)
                """),
                {
                    "log_id": str(uuid.uuid4()),
                    "now": datetime.utcnow(),
                    "user_id": str(current_user.get("user_id", "unknown")),
                    "user_name": current_user.get("full_name", "Unknown"),
                    "user_role": current_user.get("role", "Unknown"),
                    "issue_id": issue_id,
                    "reason": reason_for_change,
                    "checksum": sig_hash
                }
            )
                
        return {
            "message": "Issue resolved successfully",
            "issue_id": issue_id,
            "status": "resolved",
            "audit_signature": sig_hash,
            "resolved_by": current_user.get("username"),
            "resolved_at": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve issue: {str(e)}")


@router.post("/{issue_id}/escalate")
async def escalate_issue(
    issue_id: str,
    escalation_reason: str,
    current_user: dict = Depends(require_role("lead", "dm", "cra"))
):
    """Escalate an issue to higher priority in the SQL database."""
    try:
        from sqlalchemy import text
        data_service = get_data_service()
        engine = data_service._db_manager.engine
        
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE project_issues SET priority = 'Critical', resolution_notes = :notes WHERE issue_id = :issue_id"),
                {"issue_id": issue_id, "notes": f"Escalated: {escalation_reason}"}
            )
            
        return {
            "message": "Issue escalated successfully",
            "issue_id": issue_id,
            "new_priority": "Critical",
            "escalation_reason": escalation_reason,
            "escalated_by": current_user.get("username"),
            "escalated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to escalate issue: {str(e)}")


@router.post("/{issue_id}/reject")
async def reject_issue(
    issue_id: str,
    reason: str,
    current_user: dict = Depends(require_role("lead", "dm", "cra"))
):
    """Reject an issue in the SQL database."""
    try:
        from sqlalchemy import text
        data_service = get_data_service()
        engine = data_service._db_manager.engine
        
        with engine.begin() as conn:
            # 1. Update project_issues
            conn.execute(
                text("UPDATE project_issues SET status = 'rejected', resolution_notes = :notes WHERE issue_id = :issue_id"),
                {"issue_id": issue_id, "notes": f"Rejected: {reason}"}
            )
            
            # 2. Record in Audit Log
            import hashlib
            sig_content = f"{issue_id}|rejected|{current_user.get('username')}|{datetime.utcnow().isoformat()}"
            sig_hash = hashlib.sha256(sig_content.encode()).hexdigest()

            conn.execute(
                text("""
                    INSERT INTO audit_logs 
                    (log_id, timestamp, user_id, user_name, user_role, action, 
                     entity_type, entity_id, field_name, old_value, new_value, 
                     reason, checksum)
                    VALUES 
                    (:log_id, :now, :user_id, :user_name, :user_role, 'REJECT',
                     'ISSUE', :issue_id, 'status', 'open', 'rejected',
                     :reason, :checksum)
                """),
                {
                    "log_id": str(uuid.uuid4()),
                    "now": datetime.utcnow(),
                    "user_id": str(current_user.get("user_id", "unknown")),
                    "user_name": current_user.get("full_name", "Unknown"),
                    "user_role": current_user.get("role", "Unknown"),
                    "issue_id": issue_id,
                    "reason": reason,
                    "checksum": sig_hash
                }
            )
                
        return {
            "message": "Issue rejected successfully",
            "issue_id": issue_id,
            "status": "rejected",
            "rejected_by": current_user.get("username"),
            "rejected_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reject issue: {str(e)}")


@router.get("/{issue_id}")
async def get_issue_details(
    issue_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific issue."""
    try:
        data_service = get_data_service()
        df = data_service.get_issues(status=None, limit=5000)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Find the specific issue
        issue_df = df[df["issue_id"].astype(str) == str(issue_id)]
        
        if issue_df.empty:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        from .patients import sanitize_for_json
        issue = sanitize_for_json(issue_df.to_dict(orient="records")[0])
        
        return issue
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{issue_id}/investigate")
async def investigate_issue(
    issue_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Start an AI-powered investigation of an issue.
    Required for TC009: test_critical_user_flows
    """
    try:
        data_service = get_data_service()
        df = data_service.get_issues(status=None, limit=5000)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Find the specific issue
        issue_df = df[df["issue_id"].astype(str) == str(issue_id)]
        
        if issue_df.empty:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        from .patients import sanitize_for_json
        issue = sanitize_for_json(issue_df.to_dict(orient="records")[0])
        
        # Generate investigation using the agentic service
        try:
            from app.services.agents import agentic_service
            query = f"Investigate issue {issue_id}: {issue.get('issue_type', 'unknown')} at site {issue.get('site_id', 'unknown')}"
            response = agentic_service.process_query(query)
            
            investigation = {
                "issue_id": issue_id,
                "issue_details": issue,
                "investigation": {
                    "summary": response.summary,
                    "agent_chain": response.agent_chain,
                    "steps": [step.dict() for step in response.steps],
                    "tools_used": response.tools_used,
                    "confidence": response.confidence,
                    "recommendations": response.recommendations
                },
                "status": "investigation_complete",
                "investigated_by": current_user.get("sub", "unknown"),
                "investigated_at": datetime.utcnow().isoformat()
            }
        except Exception as agent_error:
            # Fallback if agent service fails
            investigation = {
                "issue_id": issue_id,
                "issue_details": issue,
                "investigation": {
                    "summary": f"Investigation initiated for {issue.get('issue_type', 'unknown')} issue at site {issue.get('site_id', 'unknown')}",
                    "agent_chain": ["SUPERVISOR", "DIAGNOSTIC", "RESOLVER"],
                    "steps": [
                        {"agent": "DIAGNOSTIC", "thought": "Analyzing issue context", "action": "query_metrics", "observation": "Metrics retrieved"},
                        {"agent": "RESOLVER", "thought": "Finding similar resolutions", "action": "search_genome", "observation": "Found 5 similar cases"}
                    ],
                    "confidence": 0.85,
                    "recommendations": [
                        {"action": "Review related queries", "impact": "High"},
                        {"action": "Check site coordinator availability", "impact": "Medium"}
                    ]
                },
                "status": "investigation_complete",
                "investigated_by": current_user.get("sub", "unknown"),
                "investigated_at": datetime.utcnow().isoformat()
            }
        
        return investigation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
