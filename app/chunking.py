import re

MAX_BYTES = 8000  # safely under Chroma's 16384 limit

def trim_to_bytes(text, max_bytes=MAX_BYTES):
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


def split_sentences(text):
    # naive sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

def chunk_texts(text, max_tokens=500):
    # simple token approx using words
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0
    for s in sentences:
        l = len(s.split())
        if current_len + l > max_tokens and current:
            chunk = ' '.join(current).strip()
            chunks.append(trim_to_bytes(chunk))
            current = [s]
            current_len = l
        else:
            current.append(s)
            current_len += l
    if current:
        chunk = ' '.join(current).strip()
        chunks.append(trim_to_bytes(chunk))
    return chunks
