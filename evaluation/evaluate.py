import sys
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.embeddings import EmbeddingModel
from app.db_client import ChromaClient
from app.rag import RAGSystem


# ---------------- CONFIG ----------------

TOP_K = 5

TEST_SET = [
    {
        "query": "whats is chaitanya's highest educational degree qualification?",
        "relevant_sources": ["Chaitanya Shinde Game Dev Resume.pdf"],
        "expected_answer": "Master of Science in Information Technology (MSc.IT)"
    },
    {
        "query": "what college is chaitanya pursuing his masters at?",
        "relevant_sources": ["Chaitanya Shinde Game Dev Resume.pdf"],
        "expected_answer": "Somaiya Vidhyavihar University"
    },
]


# ---------------- METRICS ----------------

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k if k else 0


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant) if relevant else 0


def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------- SEMANTIC SIMILARITY ----------------

sim_model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(a, b):
    embeddings = sim_model.encode([a, b])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


#exact match case

def exact_match(expected, generated):
    """
    Returns 1 if expected answer appears in generated answer
    (case insensitive, ignores punctuation and extra spaces)
    """

    # normalize text
    expected_clean = re.sub(r'\s+', ' ', expected.lower()).strip()
    generated_clean = re.sub(r'\s+', ' ', generated.lower()).strip()

    return int(expected_clean in generated_clean)

# ---------------- MAIN ----------------

def main():

    print("\n--- RAG System Evaluation ---\n")

    # Initialize models
    embed = EmbeddingModel(os.getenv("EMBEDDING_MODEL"))
    CHROMA_DIR = os.path.abspath(os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))

    db = ChromaClient({"persist_directory": CHROMA_DIR})
    db.ensure_collection()

    collection = db.get_user_collection("111835525617439167694")
    print("Vector count:", collection.count())
    rag = RAGSystem(embed_model=embed, db_client=db)

    # Metrics accumulators
    total_p = 0
    total_r = 0
    total_mrr = 0
    total_similarity = 0
    total_exact = 0
    total_latency = 0

    for sample in TEST_SET:

        query = sample["query"]
        relevant = [os.path.basename(r) for r in sample["relevant_sources"]]
        expected_answer = sample["expected_answer"]

        print("\n-------------------------------------")
        print("Query:", query)

        # Run full RAG pipeline
        start = time.time()

        result = rag.answer(
            question=query,
            google_id="111835525617439167694",
            top_k=TOP_K,
            model="gemini"
        )

        latency = time.time() - start
        total_latency += latency
        #print("Result is: ", result)
        generated_answer = result["answer"]

        retrieved_sources = [
            os.path.basename(s["filename"])
            for s in result.get("sources", [])
        ]

        # ---------------- RETRIEVAL METRICS ----------------

        p = precision_at_k(retrieved_sources, relevant, TOP_K)
        r = recall_at_k(retrieved_sources, relevant, TOP_K)
        rr = reciprocal_rank(retrieved_sources, relevant)

        total_p += p
        total_r += r
        total_mrr += rr

        # ---------------- ANSWER METRICS ----------------

        similarity = semantic_similarity(expected_answer, generated_answer)
        total_similarity += similarity

        exact = exact_match(expected_answer, generated_answer)
        total_exact += exact

        # ---------------- PRINT RESULTS ----------------

        print("Expected sources:", relevant)
        print("Retrieved sources:", retrieved_sources)

        print(f"Precision@{TOP_K}: {p:.3f}")
        print(f"Recall@{TOP_K}: {r:.3f}")
        print(f"RR: {rr:.3f}")

        print("\nExpected Answer:")
        print(expected_answer)

        print("\nGenerated Answer:")
        print(generated_answer[:400])

        print(f"\nSemantic Similarity: {similarity:.3f}")
        print(f"Exact Match: {exact}")
        print(f"Latency: {latency:.2f}s")

    # ---------------- FINAL RESULTS ----------------

    n = len(TEST_SET)

    print("\n\n========== FINAL RESULTS ==========\n")

    print("Retrieval Metrics")
    print("-----------------")
    print(f"Mean Precision@{TOP_K}: {total_p / n:.3f}")
    print(f"Mean Recall@{TOP_K}: {total_r / n:.3f}")
    print(f"MRR: {total_mrr / n:.3f}")

    print("\nAnswer Metrics")
    print("-----------------")
    print(f"Exact Match Rate: {total_exact / n:.3f}")
    print(f"Avg Semantic Similarity: {total_similarity / n:.3f}")

    print("\nPerformance")
    print("-----------------")
    print(f"Average Latency: {total_latency / n:.2f}s")

    print("\n===================================\n")


if __name__ == "__main__":
    main()