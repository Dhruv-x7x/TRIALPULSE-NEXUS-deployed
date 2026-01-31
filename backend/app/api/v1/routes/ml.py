"""
ML Governance Routes
====================
ML model management, approval, and monitoring endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from datetime import datetime

from app.models.schemas import MLModelListResponse, MLModelApproveRequest
from app.core.security import get_current_user, require_role
from app.services.database import get_data_service

router = APIRouter()


@router.get("/models", response_model=MLModelListResponse)
async def list_ml_models(
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all ML models with optional filters."""
    try:
        data_service = get_data_service()
        df = data_service.get_ml_models()
        
        # Apply filters
        if not df.empty:
            if status and "status" in df.columns:
                df = df[df["status"] == status]
            if model_type and "model_type" in df.columns:
                df = df[df["model_type"] == model_type]
        
        # Convert datetime columns to ISO format strings
        records = df.to_dict(orient="records")
        for record in records:
            if "trained_at" in record and record["trained_at"] is not None:
                if hasattr(record["trained_at"], "isoformat"):
                    record["trained_at"] = record["trained_at"].isoformat()
            if "deployed_at" in record and record["deployed_at"] is not None:
                if hasattr(record["deployed_at"], "isoformat"):
                    record["deployed_at"] = record["deployed_at"].isoformat()
        
        return MLModelListResponse(
            models=records,
            total=len(records)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_ml_model(
    model_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get single ML model details."""
    try:
        data_service = get_data_service()
        df = data_service.get_ml_models()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_df = df[df["version_id"] == model_id]
        
        if model_df.empty:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model_df.to_dict(orient="records")[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_ml_summary(
    current_user: dict = Depends(get_current_user)
):
    """Get ML governance summary."""
    try:
        data_service = get_data_service()
        df = data_service.get_ml_models()
        
        if df.empty:
            return {
                "total_models": 0,
                "by_status": {},
                "by_type": {},
                "deployed_count": 0,
                "pending_approval": 0
            }
        
        # Count by status
        status_counts = {}
        if "status" in df.columns:
            status_counts = df["status"].value_counts().to_dict()
        
        # Count by type
        type_counts = {}
        if "model_type" in df.columns:
            type_counts = df["model_type"].value_counts().to_dict()
        
        # Deployed models
        deployed = 0
        if "status" in df.columns:
            deployed = int((df["status"] == "deployed").sum())
        
        # Pending approval
        pending = 0
        if "status" in df.columns:
            pending = int((df["status"] == "pending_approval").sum())
        
        return {
            "total_models": len(df),
            "by_status": status_counts,
            "by_type": type_counts,
            "deployed_count": deployed,
            "pending_approval": pending
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/approve")
async def approve_model(
    model_id: int,
    request: MLModelApproveRequest,
    current_user: dict = Depends(require_role("lead", "executive"))
):
    """Approve an ML model for deployment."""
    # In production, this would update the database
    return {
        "message": "Model approved successfully",
        "model_id": model_id,
        "status": "approved",
        "approved_by": request.approved_by,
        "notes": request.notes,
        "approved_at": datetime.utcnow().isoformat()
    }


@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: int,
    current_user: dict = Depends(require_role("lead", "executive"))
):
    """Deploy an approved ML model."""
    return {
        "message": "Model deployed successfully",
        "model_id": model_id,
        "status": "deployed",
        "deployed_by": current_user.get("username"),
        "deployed_at": datetime.utcnow().isoformat()
    }


@router.post("/models/{model_id}/retire")
async def retire_model(
    model_id: int,
    reason: Optional[str] = None,
    current_user: dict = Depends(require_role("lead", "executive"))
):
    """Retire an ML model."""
    return {
        "message": "Model retired successfully",
        "model_id": model_id,
        "status": "retired",
        "reason": reason,
        "retired_by": current_user.get("username"),
        "retired_at": datetime.utcnow().isoformat()
    }


@router.get("/drift-reports")
async def get_drift_reports(
    model_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get model drift reports from the database."""
    try:
        data_service = get_data_service()
        df = data_service.get_drift_reports(model_id)
        
        if df.empty:
            # Fallback to demo data if no reports in DB
            return _get_demo_drift_reports(model_id)
            
        records = df.to_dict(orient="records")
        for record in records:
            if "checked_at" in record and record["checked_at"] is not None:
                if hasattr(record["checked_at"], "isoformat"):
                    record["checked_at"] = record["checked_at"].isoformat()
        
        return {
            "drift_reports": records,
            "total": len(records)
        }
    except Exception as e:
        # Fallback for demo
        return _get_demo_drift_reports(model_id)


def _get_demo_drift_reports(model_id: Optional[str] = None):
    """Helper to return demo drift data."""
    now = datetime.utcnow()
    drift_reports = [
        {
            "report_id": f"DEMO-{i}",
            "model_id": m[0],
            "model_name": m[0].replace('_', ' ').title(),
            "drift_score": 0.02 + (i * 0.01),
            "threshold": 0.10,
            "status": "normal",
            "baseline_accuracy": 94.0,
            "current_accuracy": 93.8,
            "checked_at": (now - timedelta(days=i)).isoformat() + "Z",
            "recommendations": "No action needed.",
            "retrain_recommended": False
        }
        for i, m in enumerate([
            ("risk_classifier", "v9.0"),
            ("issue_detector", "v3.0"),
            ("anomaly_detector", "v2.0"),
            ("resolution_predictor", "v3.0"),
            ("site_ranker", "v2.0")
        ])
    ]
    if model_id:
        drift_reports = [r for r in drift_reports if r["model_id"] == model_id]
    return {
        "drift_reports": drift_reports,
        "total": len(drift_reports)
    }




@router.get("/audit-log")
async def get_audit_log(
    model_id: Optional[int] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get ML model audit log."""
    # Mock audit log
    # In production, this would come from the database
    return {
        "audit_log": [
            {
                "log_id": 1,
                "model_id": model_id or 1,
                "action": "model_trained",
                "user": "system",
                "timestamp": datetime.utcnow().isoformat(),
                "details": "Model trained with 50000 samples"
            },
            {
                "log_id": 2,
                "model_id": model_id or 1,
                "action": "model_approved",
                "user": "lead",
                "timestamp": datetime.utcnow().isoformat(),
                "details": "Approved for production deployment"
            }
        ],
        "total": 2
    }
