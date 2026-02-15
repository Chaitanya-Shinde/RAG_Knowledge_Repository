import sys
import os
import pprint
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# allow imports from app/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.embeddings import EmbeddingModel
from app.db_client import ChromaClient


# ---------------- CONFIG ----------------

TOP_K = 5

# Add your evaluation queries here
TEST_SET = [
    {
        "query": "What's the most ordered food in canteen?",
        "relevant_sources": ["canteen_shop_data.csv"]
    },
    {
        "query":"List topics of unit 1 aml syllabus",
        "relevant_sources": ["syllabus aml.pdf"]
    },
    {
        "query":"List questions in aml questions",
        "relevant_sources": ["AML questions.txt"]
    }
    
]


# ---------------- METRICS ----------------

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant) if relevant else 0


def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------- MAIN ----------------

def main():
    print(os.getenv("EMBEDDING_MODEL"))
    embed = EmbeddingModel(os.getenv("EMBEDDING_MODEL"))
    CHROMA_DIR = os.path.abspath(os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))
    db = ChromaClient({"persist_directory": CHROMA_DIR})

    db.ensure_collection()

    total_p = 0
    total_r = 0
    total_mrr = 0

    print("\n--- RAKR Evaluation ---\n")

    print("Collections:", db.client.list_collections())
    print("COLLECTION COUNT:", db.collection.count())
    print("CHROMA PATH:", os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))




    for sample in TEST_SET:
        query = sample["query"]
        relevant = [os.path.basename(r) for r in sample["relevant_sources"]]

        q_emb = embed.embed_query(query)
        res = db.query(q_emb, n=TOP_K)

        
        # print("\nRAW CHROMA RESPONSE:")
        # pprint.pprint(res)
        # print("---- END RAW ----\n")


        retrieved_sources = []

        if res and "metadatas" in res:
            metas = res["metadatas"]

            # flatten in case of nested list
            if metas and isinstance(metas[0], list):
                metas = metas[0]

            for m in metas:
                src = m.get("source")
                if src:
                    src = os.path.basename(src)
                    if src not in retrieved_sources:
                        retrieved_sources.append(src)


        p = precision_at_k(retrieved_sources, relevant, TOP_K)
        r = recall_at_k(retrieved_sources, relevant, TOP_K)
        rr = reciprocal_rank(retrieved_sources, relevant)

        total_p += p
        total_r += r
        total_mrr += rr

        print(f"Query: {query}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Expected: {relevant}")
        print(f"Precision@{TOP_K}: {p:.3f}")
        print(f"Recall@{TOP_K}: {r:.3f}")
        print(f"RR: {rr:.3f}")
        print("-" * 40)

    n = len(TEST_SET)

    print("\n=== FINAL METRICS ===")
    print(f"Mean Precision@{TOP_K}: {total_p / n:.3f}")
    print(f"Mean Recall@{TOP_K}: {total_r / n:.3f}")
    print(f"MRR: {total_mrr / n:.3f}")
    print("====================\n")


if __name__ == "__main__":
    main()
