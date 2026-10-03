"""FastAPI routes for the new AdaFace/SCRFD/FAISS attendance system.

    POST   /api/users              register with images (multipart: name, employee_id, images[])
    GET    /api/users               list enrolled users + their stored face thumbnails
    PATCH  /api/users/{id}          rename a user (json: {"name": ...})
    DELETE /api/users/{id}          remove a user and all their embeddings
    POST   /api/users/{id}/faces    add another angle (multipart: image)
    DELETE /api/faces/{id}          remove a single stored angle
    GET    /api/attendance?date=    today's (or a given date's) attendance log
    POST   /api/recognize           identify one photo, optionally punch (multipart: image, source, punch)

All routes require the shared demo token (app/web/auth.py), same as the rest
of the app.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.attendance import enrollment
from app.web.auth import require_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


@router.post("/users")
async def create_user(name: str = Form(...), employee_id: str = Form(...), images: list[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    payload = [(img.filename or "upload.jpg", await img.read()) for img in images]
    try:
        user, results = enrollment.register_user(name=name, employee_id=employee_id, images=payload)
    except enrollment.DuplicateEmployeeId:
        raise HTTPException(status_code=409, detail=f"Employee ID '{employee_id}' is already registered")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "user": {"id": user.id, "name": user.name, "employee_id": user.employee_id},
        "images": [r.__dict__ for r in results],
    }


@router.get("/users")
def get_users():
    return enrollment.list_users()


class RenameUserRequest(BaseModel):
    name: str


@router.patch("/users/{user_id}")
def rename_user(user_id: int, body: RenameUserRequest):
    try:
        return enrollment.rename_user(user_id, body.name)
    except enrollment.UserNotFound:
        raise HTTPException(status_code=404, detail="User not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/users/{user_id}")
def remove_user(user_id: int):
    try:
        enrollment.delete_user(user_id)
    except enrollment.UserNotFound:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@router.post("/users/{user_id}/faces")
async def add_face(user_id: int, image: UploadFile = File(...)):
    try:
        result = enrollment.add_face(user_id, image.filename or "upload.jpg", await image.read())
    except enrollment.UserNotFound:
        raise HTTPException(status_code=404, detail="User not found")
    if not result.accepted:
        raise HTTPException(status_code=422, detail=result.reason)
    return result.__dict__


@router.delete("/faces/{face_id}")
def remove_face(face_id: int):
    enrollment.delete_face(face_id)
    return {"ok": True}


@router.get("/attendance")
def get_attendance(date: str | None = None):
    return enrollment.list_attendance(date)


@router.post("/recognize")
def recognize(
    image: UploadFile = File(...),
    source: str = Form("mobile"),
    punch: bool = Form(True),
):
    try:
        return enrollment.recognize_and_punch(image.file.read(), source=source, punch=punch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
