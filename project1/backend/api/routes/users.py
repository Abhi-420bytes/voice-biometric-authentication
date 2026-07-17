"""
User management endpoints.

GET    /users               — list all enrolled users
GET    /users/{username}    — enrollment status for a user
DELETE /users/{username}    — delete a user's enrollment
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from backend.api.schemas      import UserListResponse, UserStatus, DeleteResponse
from backend.api.dependencies import get_ti_pipeline, get_td_pipeline

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("", response_model=UserListResponse)
def list_users():
    pipeline = get_ti_pipeline()
    users    = pipeline.verifier.store.list_users("resemblyzer")
    return UserListResponse(users=users, count=len(users))


@router.get("/{username}", response_model=UserStatus)
def user_status(username: str):
    ti = get_ti_pipeline()
    td = get_td_pipeline()
    enrolled_ti = ti.is_enrolled(username)
    enrolled_td = td.is_enrolled(username)
    if not enrolled_ti and not enrolled_td:
        raise HTTPException(status_code=404,
                            detail=f"User '{username}' not found.")
    return UserStatus(
        username                   = username,
        enrolled_text_independent  = enrolled_ti,
        enrolled_text_dependent    = enrolled_td,
    )


@router.delete("/{username}", response_model=DeleteResponse)
def delete_user(username: str):
    ti = get_ti_pipeline()
    store = ti.verifier.store

    if not store.is_enrolled(username, "resemblyzer"):
        raise HTTPException(status_code=404,
                            detail=f"User '{username}' not found.")
    try:
        store.delete_user(username, "resemblyzer")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    return DeleteResponse(
        username = username,
        deleted  = True,
        message  = f"User '{username}' removed from all enrollments.",
    )
