# app/ingestion/parser.py

from pathlib import Path
from pypdf import PdfReader
import docx
import pandas as pd
from PIL import Image
import pytesseract
import io
import logging

LOG = logging.getLogger("rakr.parser")


def parse_file(file_stream, filename):
    ext = Path(filename).suffix.lower()

    try:
        if ext == ".pdf":
            reader = PdfReader(file_stream)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            content = "\n".join(text)

        elif ext in [".docx", ".doc"]:
            document = docx.Document(file_stream)
            content = "\n".join([p.text for p in document.paragraphs])

        elif ext in [".txt", ".md"]:
            content = file_stream.read().decode("utf-8", errors="ignore")

        elif ext == ".csv":
            try:
                df = pd.read_csv(file_stream, encoding="utf-8", engine="python", on_bad_lines="skip")
            except Exception:
                file_stream.seek(0)
                df = pd.read_csv(file_stream, encoding="latin1", engine="python", on_bad_lines="skip")
            content = df.to_string()

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_stream, engine="openpyxl")
            content = df.to_string()

        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
            image = Image.open(file_stream)
            content = pytesseract.image_to_string(image)

        else:
            content = file_stream.read().decode("utf-8", errors="ignore")

    except Exception as e:
        LOG.exception("Error parsing %s: %s", filename, e)
        content = ""

    return content