from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from app.drive.drive_service import get_drive_service, get_or_create_rag_folder
import os

router = APIRouter()

flow_store = {}
user_store = {}

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

    drive_service = get_drive_service(
        credentials.token,
        credentials.refresh_token,
        os.getenv("GOOGLE_CLIENT_ID"),
        os.getenv("GOOGLE_CLIENT_SECRET")
    )

    folder_id = get_or_create_rag_folder(drive_service)

    user_store["current_user"] = {
    "access_token": credentials.token,
    "refresh_token": credentials.refresh_token,
    "folder_id": folder_id
    }