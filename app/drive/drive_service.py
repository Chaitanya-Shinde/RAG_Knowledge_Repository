from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseUpload
import io


def get_drive_service(access_token, refresh_token, client_id, client_secret):
    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
    )

    service = build("drive", "v3", credentials=creds)
    return service


def get_or_create_rag_folder(service):
    query = "name='RAG-Knowledge' and mimeType='application/vnd.google-apps.folder' and trashed=false"

    results = service.files().list(q=query, spaces="drive").execute()
    items = results.get("files", [])

    if items:
        return items[0]["id"]

    # Create folder
    folder_metadata = {
        "name": "RAG-Knowledge",
        "mimeType": "application/vnd.google-apps.folder",
    }

    folder = service.files().create(body=folder_metadata).execute()
    return folder.get("id")


def upload_file_to_drive(service, file, filename, folder_id):
    file_metadata = {
        "name": filename,
        "parents": [folder_id]
    }

    file_stream = io.BytesIO(file.file.read())
    print(file_metadata)
    media = MediaIoBaseUpload(
        file_stream,
        mimetype=file.content_type,
        resumable=True
    )

    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    return uploaded_file.get("id")