# app/ingestion/parser.py
import os
from pathlib import Path
from pypdf import PdfReader
import docx
import pandas as pd
from PIL import Image
import pytesseract
import logging

LOG = logging.getLogger("rakr.parser")

def parse_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except Exception as e:
            LOG.debug("PDF page parse error: %s", e)
            text.append("")
    return "\n".join(text)

def parse_docx(path):
    try:
        doc = docx.Document(path)
        texts = [p.text for p in doc.paragraphs]
        return "\n".join(texts)
    except Exception as e:
        LOG.exception("DOCX parse failed: %s", e)
        return ""

def parse_txt(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        LOG.exception("TXT parse failed: %s", e)
        return ""

def parse_csv(path):
    """
    Robust CSV parsing:
    - First try utf-8 with pandas (engine='python', on_bad_lines='skip').
    - If that fails, try latin1.
    - If pandas still fails (binary or very malformed), fall back to plain text read.
    """
    try:
        # engine='python' + on_bad_lines helps with messy CSVs
        df = pd.read_csv(path, encoding='utf-8', engine='python', on_bad_lines='skip')
        return df.to_string()
    except Exception as e1:
        LOG.warning("CSV parse utf-8 failed (%s). Trying latin1...", e1)
        try:
            df = pd.read_csv(path, encoding='latin1', engine='python', on_bad_lines='skip')
            return df.to_string()
        except Exception as e2:
            LOG.warning("CSV parse latin1 failed (%s). Falling back to raw text read.", e2)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e3:
                LOG.exception("CSV fallback read failed: %s", e3)
                return ""

def parse_xlsx(path):
    try:
        df = pd.read_excel(path, engine='openpyxl')
        return df.to_string()
    except Exception as e:
        LOG.exception("XLSX parse failed: %s", e)
        return ""

def parse_image(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        LOG.exception("Image OCR failed: %s", e)
        return ""

def parse_file(path):
    ext = Path(path).suffix.lower()
    try:
        if ext == '.pdf':
            content = parse_pdf(path)
        elif ext in ['.docx', '.doc']:
            content = parse_docx(path)
        elif ext in ['.txt', '.md']:
            content = parse_txt(path)
        elif ext in ['.csv']:
            content = parse_csv(path)
        elif ext in ['.xlsx', '.xls']:
            content = parse_xlsx(path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
            content = parse_image(path)
        else:
            # fallback: try plain text
            content = parse_txt(path)
    except Exception as e:
        LOG.exception("Unexpected error parsing file %s: %s", path, e)
        content = ""
    return {"content": content, "source": path}
