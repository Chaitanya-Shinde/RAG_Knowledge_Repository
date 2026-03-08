from fastapi import Request, HTTPException
from app.auth.jwt_handler import verify_token
from app.db.mongo import users_collection

def get_current_user(request: Request):
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    google_id = payload.get("google_id")

    user = users_collection.find_one({"google_id": google_id})

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user