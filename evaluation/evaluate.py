import sys
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

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
MODELS_TO_EVALUATE = ["llama3.2:1b-instruct-q4_K_M", "deepseek"]
USER_ID = "111835525617439167694"  # Test user ID

TEST_SET = [
  {
    "query": "What degree is Chaitanya currently pursuing?",
    "expected_answer": "Msc IT",
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What university is Chaitanya pursuing MSc IT from?",
    "expected_answer": "Somaiya Vidhyavihar University",
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What is Chaitanya's bachelor's degree?",
    "expected_answer": "Bsc. CS",
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What project involves gesture-based mouse control?",
    "expected_answer": "Gesture mouse controller",
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "Does chaitanya know python? Answer Yes or No",
    "expected_answer": "Yes",
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },

  {
    "query": "What is React framework mentioned in the front end resume? Answer Yes or No",
    "expected_answer": "Yes",
    "relevant_sources": ["Front End Resume.pdf"],
    "difficulty": "medium",
    "type": "fact"
  },
  {
    "query": "What university is listed in the front end resume?",
    "expected_answer": "San Jose State University",
    "relevant_sources": ["Front End Resume.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },

  {
    "query": "What are the five core values of the Tata Code of Conduct? Answer only the core values",
    "expected_answer": "Integrity, Excellence, Pioneering, Unity, Responsibility",
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "Does Tata tolerate bribery or corruption?",
    "expected_answer": "does not tolerate",
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What does the Tata Code expect honesty? Answer Yes or No",
    "expected_answer": "Yes",
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What type of learning is KNN?",
    "expected_answer": "Supervised learning",
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "easy",
    "type": "conceptual"
  },
  {
    "query": "What is classification in machine learning?",
    "expected_answer": "Assigning objects into categories",
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What is regression used for in machine learning?",
    "expected_answer": "predicting numerical outcomes",
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What technique is used to evaluate machine learning models?",
    "expected_answer": "Cross-validation",
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "medium",
    "type": "conceptual"
  },

  {
    "query": "What is the highest salary in the employee dataset?",
    "expected_answer": "215000",
    "relevant_sources": ["Employee_data.csv"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "What is the salary of Robert Rivera?",
    "expected_answer": "205000",
    "relevant_sources": ["Employee_data.csv"],
    "difficulty": "easy",
    "type": "numeric"
  },
  {
    "query": "What department does employee Silvia Gibson belong to?",
    "expected_answer": "Sales",
    "relevant_sources": ["employeeData.json"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "How many employees belong to department Finance?",
    "expected_answer": "1596",
    "relevant_sources": ["employeeData.json"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "What is the lowest salary in the employee dataset?",
    "expected_answer": "25000",
    "relevant_sources": ["employeeData.json"],
    "difficulty": "medium",
    "type": "numeric"
  },

  {
    "query": "What is the date mentioned in the society notice?",
    "expected_answer": "22 Jan 2023",
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "When is the special general meeting scheduled?",
    "expected_answer": "Sunday 22 Jan 2023 at 10 AM",
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "hard",
    "type": "ocr"
  },
  {
    "query": "What is the society name?",
    "expected_answer": "RUNWAL GARDEN CITY C-1 & C-2 CO-OP. HSG. SOCIETY LTD",
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "easy",
    "type": "ocr"
  },
  {
    "query": "What is the agenda of the society meeting?",
    "expected_answer": "findings of the Structural Audit Report",
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },

  {
    "query": "What is the grand total amount on the receipt?",
    "expected_answer": "860",
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "easy",
    "type": "ocr"
  },
  {
    "query": "How many items were purchased according to the receipt?",
    "expected_answer": "5",
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What is the subtotal amount on the receipt?",
    "expected_answer": "819.04",
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What CGST percentage is applied on the receipt?",
    "expected_answer": "2.5%",
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },

  {
    "query": "What is normalization in DBMS?",
    "expected_answer": "Reducing redundancy",
    "relevant_sources": ["DBMS_notes.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What is a primary key in DBMS?",
    "expected_answer": "unique identifier",
    "relevant_sources": ["DBMS_notes.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },

  {
    "query": "Compare chaitanya's skills mentioned in both resumes",
    "expected_answer": "JavaScript and Node",
    "relevant_sources": [
      "Chaitanya Shinde Resume updated.pdf",
      "Front End Resume.pdf"
    ],
    "difficulty": "hard",
    "type": "multi_document"
  },

  {
    "query": "What is the CEO's private password?",
    "expected_answer": "not found",
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  },
  {
    "query": "What is the secret internal financial data?",
    "expected_answer": "not found",
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  },
  {
    "query": "What is the personal bank account number of employees?",
    "expected_answer": "not found",
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  }
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


def f1_at_k(retrieved, relevant, k):
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_average_precision(retrieved, relevant):
    """Calculate Mean Average Precision for a single query"""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(retrieved):
        if item in relevant_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    return sum_precisions / len(relevant) if relevant else 0.0


# ---------------- SEMANTIC SIMILARITY ----------------

sim_model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(a, b):
    embeddings = sim_model.encode([a, b])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def exact_match(expected, generated):
    """
    Returns 1 if expected answer appears in generated answer
    (case insensitive, ignores punctuation and extra spaces)
    """
    expected_clean = re.sub(r'\s+', ' ', expected.lower()).strip()
    generated_clean = re.sub(r'\s+', ' ', generated.lower()).strip()
    return int(expected_clean in generated_clean)


# ---------------- LLM-BASED EVALUATION ----------------

def evaluate_faithfulness_with_llm(answer, context, rag_system, judge_model="deepseek"):
    """
    Use LLM to evaluate if the answer is faithful to the context.
    Returns a score from 0-1.
    """
    prompt = f"""
    Evaluate if the following answer is faithful to the provided context.
    Faithfulness means the answer does not contradict the context and all claims are supported by the context.

    Context:
    {context[:2000]}  # Limit context length

    Answer:
    {answer}

    Rate the faithfulness on a scale of 0-1, where:
    - 1.0: Completely faithful, all information is supported by context
    - 0.5: Mostly faithful but has some unsupported claims
    - 0.0: Not faithful, contradicts context or has major unsupported claims

    Respond with only a number between 0 and 1.
    """

    try:
        time.sleep(6)  # Rate limit prevention
        response = rag_system._call_llm_raw(prompt, judge_model, max_tokens=10)
        score = float(response.strip())
        return max(0.0, min(1.0, score))  # Clamp to 0-1
    except:
        return 0.5  # Default to neutral


def evaluate_relevance_with_llm(answer, query, rag_system, judge_model="deepseek"):
    """
    Use LLM to evaluate if the answer is relevant to the query.
    Returns a score from 0-1.
    """
    prompt = f"""
    Evaluate if the following answer is relevant to the query.

    Query: {query}

    Answer: {answer}

    Rate the relevance on a scale of 0-1, where:
    - 1.0: Highly relevant, directly addresses the query
    - 0.5: Somewhat relevant but misses key aspects
    - 0.0: Not relevant to the query

    Respond with only a number between 0 and 1.
    """

    try:
        time.sleep(6)  # Rate limit prevention
        response = rag_system._call_llm_raw(prompt, judge_model, max_tokens=10)
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_context_relevance_with_llm(context, query, rag_system, judge_model="deepseek"):
    """
    Use LLM to evaluate if the retrieved context is relevant to the query.
    Returns a score from 0-1.
    """
    prompt = f"""
    Evaluate if the following context is relevant to the query.

    Query: {query}

    Context:
    {context[:2000]}

    Rate the relevance on a scale of 0-1, where:
    - 1.0: Highly relevant, contains information directly related to the query
    - 0.5: Somewhat relevant but may contain tangential information
    - 0.0: Not relevant to the query

    Respond with only a number between 0 and 1.
    """

    try:
        time.sleep(6)  # Rate limit prevention
        response = rag_system._call_llm_raw(prompt, judge_model, max_tokens=10)
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5


# ---------------- MAIN ----------------

def evaluate_model(rag, model_name, test_set, top_k):
    """Evaluate a single model on the test set"""
    print(f"\n{'='*50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'='*50}")

    # Metrics accumulators
    metrics = {
        'precision': 0, 'recall': 0, 'f1': 0, 'mrr': 0, 'map': 0,
        'semantic_sim': 0, 'exact_match': 0,
        'faithfulness': 0, 'relevance': 0, 'context_relevance': 0,
        'latency': 0
    }

    for i, sample in enumerate(test_set):
        query = sample["query"]
        relevant = [os.path.basename(r) for r in sample["relevant_sources"]]
        expected_answer = sample["expected_answer"]

        print(f"\n--- Query {i+1}: {query[:60]}... ---")

        # Run full RAG pipeline
        start = time.time()
        result = rag.answer(
            question=query,
            google_id=USER_ID,
            top_k=top_k,
            model=model_name
        )
        latency = time.time() - start

        generated_answer = result["answer"]
        retrieved_sources = [
            os.path.basename(s["filename"])
            for s in result.get("sources", [])
        ]

        # Get retrieved context for evaluation
        context = result.get("context", "")  # Assuming rag.answer returns context

        print(f"Expected: {expected_answer}")
        print(f"Generated: {generated_answer}")

        # Retrieval Metrics
        p = precision_at_k(retrieved_sources, relevant, top_k)
        r = recall_at_k(retrieved_sources, relevant, top_k)
        f1 = f1_at_k(retrieved_sources, relevant, top_k)
        rr = reciprocal_rank(retrieved_sources, relevant)
        map_score = mean_average_precision(retrieved_sources, relevant)

        # Answer Quality Metrics
        sem_sim = semantic_similarity(expected_answer, generated_answer)
        exact = exact_match(expected_answer, generated_answer)

        # LLM-based Metrics
        faithfulness = evaluate_faithfulness_with_llm(generated_answer, context, rag)
        relevance = evaluate_relevance_with_llm(generated_answer, query, rag)
        ctx_relevance = evaluate_context_relevance_with_llm(context, query, rag)

        # Accumulate
        for key, value in [('precision', p), ('recall', r), ('f1', f1), ('mrr', rr), ('map', map_score),
                          ('semantic_sim', sem_sim), ('exact_match', exact),
                          ('faithfulness', faithfulness), ('relevance', relevance), ('context_relevance', ctx_relevance),
                          ('latency', latency)]:
            metrics[key] += value

        # Print individual results
        print(f"Retrieved: {retrieved_sources}")
        print(f"Relevant: {relevant}")
        print(f"P@{top_k}: {p:.3f}, R@{top_k}: {r:.3f}, F1: {f1:.3f}, RR: {rr:.3f}, MAP: {map_score:.3f}")
        print(f"Semantic Sim: {sem_sim:.3f}, Exact Match: {exact}")
        print(f"Faithfulness: {faithfulness:.3f}, Relevance: {relevance:.3f}, Context Rel: {ctx_relevance:.3f}")
        print(f"Latency: {latency:.2f}s")

    # Average metrics
    n = len(test_set)
    avg_metrics = {k: v/n for k, v in metrics.items()}

    print(f"\n--- AVERAGE METRICS FOR {model_name.upper()} ---")
    print(f"Retrieval: P@{top_k}={avg_metrics['precision']:.3f}, R@{top_k}={avg_metrics['recall']:.3f}, "
          f"F1={avg_metrics['f1']:.3f}, MRR={avg_metrics['mrr']:.3f}, MAP={avg_metrics['map']:.3f}")
    print(f"Answer Quality: Semantic Sim={avg_metrics['semantic_sim']:.3f}, Exact Match={avg_metrics['exact_match']:.3f}")
    print(f"LLM Judged: Faithfulness={avg_metrics['faithfulness']:.3f}, Relevance={avg_metrics['relevance']:.3f}, "
          f"Context Rel={avg_metrics['context_relevance']:.3f}")
    print(f"Performance: Avg Latency={avg_metrics['latency']:.2f}s")

    return avg_metrics

def convert_numpy_types(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def main():
    print("\n--- RAG System Evaluation for Research Paper ---\n")

    # Initialize models
    embed = EmbeddingModel(os.getenv("EMBEDDING_MODEL"))
    CHROMA_DIR = os.path.abspath(os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))
    db = ChromaClient({"persist_directory": CHROMA_DIR})
    db.ensure_collection()

    collection = db.get_user_collection(USER_ID)
    print(f"Vector count: {collection.count()}")
    rag = RAGSystem(embed_model=embed, db_client=db)

    # Evaluate all models
    all_results = {}
    for model in MODELS_TO_EVALUATE:
        try:
            results = evaluate_model(rag, model, TEST_SET, TOP_K)
            all_results[model] = results
        except Exception as e:
            print(f"Error evaluating {model}: {e}")
            all_results[model] = {"error": str(e)}

    # Comparative Analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS ACROSS MODELS")
    print(f"{'='*60}")

    metrics_to_compare = ['precision', 'recall', 'f1', 'mrr', 'map', 'semantic_sim', 'exact_match',
                         'faithfulness', 'relevance', 'context_relevance', 'latency']

    print(f"{'Metric':<20} {' '.join(m[:8].ljust(8) for m in MODELS_TO_EVALUATE)}")
    print("-" * (20 + 9 * len(MODELS_TO_EVALUATE)))

    print(f"{'Metric':<20}", end="")

    for model in MODELS_TO_EVALUATE:
        print(f"{model[:12]:>12}", end="")

    print()

    print("-" * (20 + 12 * len(MODELS_TO_EVALUATE)))

    for metric in metrics_to_compare:

        print(f"{metric:<20}", end="")

        for model in MODELS_TO_EVALUATE:

            if model in all_results and 'error' not in all_results[model]:
                value = all_results[model][metric]
                print(f"{value:>12.3f}", end="")
            else:
                print(f"{'N/A':>12}", end="")

        print()

    clean_results = convert_numpy_types(all_results)

    
    # Save results to JSON for paper
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "config": {
                "top_k": TOP_K,
                "models": MODELS_TO_EVALUATE,
                "test_set_size": len(TEST_SET),
                "user_id": USER_ID
            },
            "results": clean_results,
            "test_queries": TEST_SET
        }, f, indent=2)

    

    # Log results to text file
    log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write("RAG System Evaluation Results\n")
        log_file.write("=" * 50 + "\n\n")
        log_file.write(f"Config:\n")
        log_file.write(f"  Top-K: {TOP_K}\n")
        log_file.write(f"  Models: {', '.join(MODELS_TO_EVALUATE)}\n")
        log_file.write(f"  Test Set Size: {len(TEST_SET)}\n")
        log_file.write(f"  User ID: {USER_ID}\n\n")
        
        for model, results in all_results.items():
            if 'error' not in results:
                log_file.write(f"Model: {model.upper()}\n")
                log_file.write("-" * 30 + "\n")
                log_file.write(f"Retrieval: P@{TOP_K}={results['precision']:.3f}, R@{TOP_K}={results['recall']:.3f}, "
                              f"F1={results['f1']:.3f}, MRR={results['mrr']:.3f}, MAP={results['map']:.3f}\n")
                log_file.write(f"Answer Quality: Semantic Sim={results['semantic_sim']:.3f}, Exact Match={results['exact_match']:.3f}\n")
                log_file.write(f"LLM Judged: Faithfulness={results['faithfulness']:.3f}, Relevance={results['relevance']:.3f}, "
                              f"Context Rel={results['context_relevance']:.3f}\n")
                log_file.write(f"Performance: Avg Latency={results['latency']:.2f}s\n\n")
            else:
                log_file.write(f"Model: {model.upper()} - Error: {results['error']}\n\n")
        
        log_file.write("Comparative Analysis\n")
        log_file.write("=" * 30 + "\n")
        log_file.write(f"{'Metric':<20} {' '.join(m[:8].ljust(8) for m in MODELS_TO_EVALUATE)}\n")
        log_file.write("-" * (20 + 9 * len(MODELS_TO_EVALUATE)) + "\n")
        for metric in metrics_to_compare:
            values = []
            for model in MODELS_TO_EVALUATE:
                if model in all_results and 'error' not in all_results[model]:
                    values.append(f"{all_results[model][metric]:.3f}")
                else:
                    values.append("N/A")
            log_file.write(metric.ljust(20) + '  ' + '  '.join(values) + "\n")
        
        log_file.write("\nEvaluation complete!\n")

    print(f"\nResults saved to evaluation_results.json and {log_filename}")
    print("\nEvaluation complete! Use these metrics in your research paper to demonstrate:")
    print("- Retrieval effectiveness (Precision, Recall, F1, MRR, MAP)")
    print("- Answer quality (Semantic Similarity, Exact Match)")
    print("- Answer faithfulness and relevance (LLM-judged)")
    print("- Context relevance to queries")
    print("- Performance differences across LLM models")


if __name__ == "__main__":
    main()