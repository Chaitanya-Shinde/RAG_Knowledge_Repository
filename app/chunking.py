import re

MAX_BYTES = 8000  # safely under Chroma's 16384 limit


def trim_to_bytes(text, max_bytes=MAX_BYTES):
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


# -------------------------
# Sentence splitter
# -------------------------

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences


# -------------------------
# Detect section headings
# -------------------------

def split_sections(text):
    """
    Split document into sections using common heading patterns.
    Works well for PDFs, tutorials, and docs.
    """

    pattern = r'\n(?=[A-Z][A-Za-z0-9\s]{3,60}\n)'

    sections = re.split(pattern, text)

    cleaned = [s.strip() for s in sections if len(s.strip()) > 50]

    return cleaned


# -------------------------
# Chunk inside sections
# -------------------------

def chunk_section(section_text, max_tokens=300):

    sentences = split_sentences(section_text)

    chunks = []
    current = []
    current_len = 0

    for s in sentences:

        l = len(s.split())

        if current_len + l > max_tokens and current:

            chunk = " ".join(current).strip()

            chunks.append(trim_to_bytes(chunk))

            current = [s]
            current_len = l

        else:

            current.append(s)
            current_len += l

    if current:
        chunk = " ".join(current).strip()
        chunks.append(trim_to_bytes(chunk))

    return chunks


# -------------------------
# Main chunking function
# -------------------------

def chunk_texts(text, max_tokens=500):

    sections = split_sections(text)

    chunks = []

    for section in sections:

        section_chunks = chunk_section(section, max_tokens=max_tokens)

        chunks.extend(section_chunks)

    return chunks