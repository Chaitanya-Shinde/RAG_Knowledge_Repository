from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse
from google_auth_oauthlib.flow import Flow
from app.drive.drive_service import get_drive_service, get_or_create_rag_folder
from app.db.mongo import users_collection
import os
from google.oauth2 import id_token
from google.auth.transport import requests
from app.auth.jwt_handler import create_access_token

router = APIRouter()

flow_store = {}


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.file",
]

REDIRECT_URI = "http://localhost:8000/auth/callback"

@router.get("/login")
def login():
    flow = Flow.from_client_secrets_file(
        "credentials.json",
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true"
    )

    flow_store[state] = flow

    return RedirectResponse(auth_url)


@router.get("/callback")
def callback(request: Request):
    state = request.query_params.get("state")
    code = request.query_params.get("code")

    flow = flow_store.get(state)

    if not flow:
        return {"error": "Invalid state"}

    flow.fetch_token(
        code=code,
        authorization_response=str(request.url)
    )

    credentials = flow.credentials

    id_info = id_token.verify_oauth2_token(
        credentials.id_token,
        requests.Request(),
        os.getenv("GOOGLE_CLIENT_ID")
    )

    google_id = id_info["sub"]
    email = id_info["email"]
    name = id_info.get("name")

    drive_service = get_drive_service(credentials.refresh_token)

    folder_id = get_or_create_rag_folder(drive_service)

    users_collection.update_one(
        {"google_id": google_id},
        {
            "$set": {
                "email": email,
                "name": name,
                "drive_folder_id": folder_id,
                "refresh_token": credentials.refresh_token,
            }
        },
        upsert=True
    )

    jwt_token = create_access_token({"google_id": google_id})

    response = RedirectResponse(url="http://localhost:5173")  # your frontend

    
    response.set_cookie(
        key="access_token",
        value=jwt_token,
        httponly=True,
        secure=False,   # set True in production (HTTPS)
        samesite="lax"
    )

    
    return response