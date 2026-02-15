import re

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
            chunks.append(' '.join(current).strip())
            current = [s]
            current_len = l
        else:
            current.append(s)
            current_len += l
    if current:
        chunks.append(' '.join(current).strip())
    return chunks
