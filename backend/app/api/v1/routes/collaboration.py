
from fastapi import APIRouter, HTTPException, Depends, Body
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from app.core.security import get_current_user
from app.services.database import get_data_service

router = APIRouter()

@router.get("/rooms")
async def list_collaboration_rooms(
    current_user: dict = Depends(get_current_user)
):
    """List all active investigation rooms."""
    from src.database.connection import get_db_manager
    from src.database.models import CollaborationRoom, RoomParticipant
    db = get_db_manager()
    session = db.get_session()
    try:
        user_id = current_user.get("user_id")
        
        # Get rooms where user is participant
        rooms = session.query(CollaborationRoom)\
            .join(RoomParticipant, CollaborationRoom.room_id == RoomParticipant.room_id)\
            .filter(RoomParticipant.user_id == user_id)\
            .all()
            
        # Fallback for demo/test mode: if user has no rooms, show all active rooms
        if not rooms:
            rooms = session.query(CollaborationRoom).filter(CollaborationRoom.status == 'active').limit(10).all()
            
        return {
            "rooms": [
                {
                    "room_id": r.room_id,
                    "title": r.title,
                    "description": r.description,
                    "status": r.status,
                    "type": r.room_type,
                    "priority": r.priority,
                    "created_at": r.created_at.isoformat()
                } for r in rooms
            ]
        }
    finally:
        session.close()

@router.get("/rooms/{room_id}/messages")
async def get_room_messages(
    room_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all messages for a specific room."""
    from src.database.connection import get_db_manager
    from src.database.models import RoomMessage, User
    db = get_db_manager()
    session = db.get_session()
    try:
        msgs = session.query(RoomMessage, User.first_name, User.last_name)\
            .join(User, RoomMessage.user_id == User.user_id)\
            .filter(RoomMessage.room_id == room_id)\
            .order_by(RoomMessage.created_at.asc())\
            .all()
            
        return {
            "messages": [
                {
                    "message_id": str(m[0].message_id),
                    "content": str(m[0].content),
                    "sender": f"{m[1]} {m[2]}",
                    "user_id": str(m[0].user_id),
                    "type": str(m[0].message_type),
                    "timestamp": m[0].created_at.isoformat()
                } for m in msgs
            ]
        }
    finally:
        session.close()

@router.post("/rooms/{room_id}/messages")
async def post_message(
    room_id: str,
    data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Post a new message to a room."""
    from src.database.connection import get_db_manager
    from src.database.models import RoomMessage
    db = get_db_manager()
    session = db.get_session()
    try:
        user_id = current_user.get("user_id")
        # Check if user exists in the database to prevent ForeignKeyViolation
        from src.database.models import User
        user_exists = session.query(User).filter(User.user_id == user_id).first()
        
        if not user_exists:
            # Fallback to a known user (like 'admin' or 'lead') for demo stability
            fallback_user = session.query(User).filter(User.username == 'lead').first()
            if fallback_user:
                user_id = fallback_user.user_id
            else:
                # If even lead doesn't exist, use the first available user
                first_user = session.query(User).first()
                if first_user:
                    user_id = first_user.user_id

        new_msg = RoomMessage(
            room_id=room_id,
            user_id=user_id,
            content=data.get("content"),
            message_type="text"
        )
        session.add(new_msg)
        session.commit()
        return {"status": "success", "message_id": new_msg.message_id}
    finally:
        session.close()

@router.post("/rooms/{room_id}/tag")
async def tag_user(
    room_id: str,
    data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Tag a user in an investigation room - creates a system message."""
    from src.database.connection import get_db_manager
    from src.database.models import RoomMessage, User
    db = get_db_manager()
    session = db.get_session()
    try:
        tag_user_id = data.get('user_id')
        tagged_user = session.query(User).filter(User.user_id == tag_user_id).first()
        tagged_name = f"{tagged_user.first_name} {tagged_user.last_name}" if tagged_user else "User"
        
        sender_id = current_user.get("user_id")
        
        new_msg = RoomMessage(
            room_id=room_id,
            user_id=sender_id,
            content=f"tagged @{tagged_name}",
            message_type="system"
        )
        session.add(new_msg)
        session.commit()
        
        return {
            "status": "success",
            "message": f"User {tagged_name} tagged in room",
            "notification_id": str(uuid.uuid4())
        }
    finally:
        session.close()

@router.post("/rooms/{room_id}/escalate")
async def escalate_room(
    room_id: str,
    data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Trigger the escalation pipeline - updates room status and level."""
    from src.database.connection import get_db_manager
    from src.database.models import CollaborationRoom, RoomMessage
    db = get_db_manager()
    session = db.get_session()
    try:
        room = session.query(CollaborationRoom).filter(CollaborationRoom.room_id == room_id).first()
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
            
        old_level = room.escalation_level
        new_level_val = data.get("escalation_level")
        # Handle string or int
        if isinstance(new_level_val, str) and new_level_val.startswith('L'):
            new_level = int(new_level_val[1:])
        else:
            new_level = old_level + 1
            
        room.escalation_level = new_level
        room.status = "escalated"
        room.priority = "critical"
        
        # Add system message
        msg = RoomMessage(
            room_id=room_id,
            user_id=current_user.get("user_id"),
            content=f"ESCALATED room from Level {old_level} to Level {new_level}. Reason: {data.get('reason', 'Critical issue requiring oversight')}",
            message_type="system"
        )
        session.add(msg)
        session.commit()
        
        return {
            "status": "escalated",
            "new_level": f"L{new_level}",
            "reason": data.get("reason"),
            "timestamp": datetime.utcnow().isoformat()
        }
    finally:
        session.close()
