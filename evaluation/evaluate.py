import sys
import os
import time
import re
import csv
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.embeddings import EmbeddingModel
from app.db_client import ChromaClient
from app.rag import RAGSystem


# ---------------- CONFIG ----------------

TOP_K = 5
MODELS_TO_EVALUATE = ["llama3.2:1b-instruct-q4_K_M", "deepseek"]
USER_ID = "111835525617439167694"

TEST_SET = [
  {
    "query": "What degree is Chaitanya currently pursuing?",
    "expected_answer": ["Msc IT", "MSc IT", "Master of Science in IT"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What university is Chaitanya pursuing MSc IT from?",
    "expected_answer": ["Somaiya Vidhyavihar University", "SVVV"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What is Chaitanya's bachelor's degree?",
    "expected_answer": ["Bsc. CS", "BSc CS", "Bachelor of Science in Computer Science"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What project involves gesture-based mouse control?",
    "expected_answer": ["Gesture mouse controller", "gesture control"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "Does chaitanya know python? Answer Yes or No",
    "expected_answer": ["Yes"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What is React framework mentioned in the front end resume? Answer Yes or No",
    "expected_answer": ["Yes"],
    "relevant_sources": ["Front End Resume.pdf"],
    "difficulty": "medium",
    "type": "fact"
  },
  {
    "query": "What university is listed in the front end resume?",
    "expected_answer": ["San Jose State University", "SJSU"],
    "relevant_sources": ["Front End Resume.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What are the five core values of the Tata Code of Conduct? Answer only the core values",
    "expected_answer": ["Integrity, Excellence, Pioneering, Unity, Responsibility", "Integrity Excellence Pioneering Unity Responsibility"],
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "Does Tata tolerate bribery or corruption?",
    "expected_answer": ["does not tolerate", "no", "does not"],
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What does the Tata Code expect honesty? Answer Yes or No",
    "expected_answer": ["Yes"],
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What type of learning is KNN?",
    "expected_answer": ["Supervised learning", "supervised"],
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "easy",
    "type": "conceptual"
  },
  {
    "query": "What is classification in machine learning?",
    "expected_answer": ["Assigning objects into categories", "assigns objects to categories"],
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What is regression used for in machine learning?",
    "expected_answer": ["predicting numerical outcomes", "predict numerical values"],
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What technique is used to evaluate machine learning models?",
    "expected_answer": ["Cross-validation", "cross validation"],
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What is the highest salary in the employee dataset?",
    "expected_answer": ["215000", "215,000"],
    "relevant_sources": ["Employee_data.csv"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "What is the salary of Robert Rivera?",
    "expected_answer": ["205000", "205,000"],
    "relevant_sources": ["Employee_data.csv"],
    "difficulty": "easy",
    "type": "numeric"
  },
  {
    "query": "What department does employee Silvia Gibson belong to?",
    "expected_answer": ["Sales"],
    "relevant_sources": ["employeeData.json"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "How many employees belong to department Finance?",
    "expected_answer": ["1596"],
    "relevant_sources": ["employeeData.json"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "What is the lowest salary in the employee dataset?",
    "expected_answer": ["25000", "25,000"],
    "relevant_sources": ["employeeData.json"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "What is the date mentioned in the society notice?",
    "expected_answer": ["22 Jan 2023", "January 22 2023"],
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "When is the special general meeting scheduled?",
    "expected_answer": ["Sunday 22 Jan 2023 at 10 AM", "22 January 2023 10 AM"],
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "hard",
    "type": "ocr"
  },
  {
    "query": "What is the society name?",
    "expected_answer": ["RUNWAL GARDEN CITY C-1 & C-2 CO-OP. HSG. SOCIETY LTD"],
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "easy",
    "type": "ocr"
  },
  {
    "query": "What is the agenda of the society meeting?",
    "expected_answer": ["findings of the Structural Audit Report", "structural audit"],
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What is the grand total amount on the receipt?",
    "expected_answer": ["860"],
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "easy",
    "type": "ocr"
  },
  {
    "query": "How many items were purchased according to the receipt?",
    "expected_answer": ["5"],
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What is the subtotal amount on the receipt?",
    "expected_answer": ["819.04", "819"],
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What CGST percentage is applied on the receipt?",
    "expected_answer": ["2.5%", "2.5", "2.5 percent"],
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What is normalization in DBMS?",
    "expected_answer": ["Reducing redundancy", "reduce redundancy"],
    "relevant_sources": ["DBMS_notes.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What is a primary key in DBMS?",
    "expected_answer": ["unique identifier", "uniquely identifies"],
    "relevant_sources": ["DBMS_notes.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "Compare chaitanya's skills mentioned in both resumes",
    "expected_answer": ["JavaScript and Node", "javascript nodejs"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf", "Front End Resume.pdf"],
    "difficulty": "hard",
    "type": "multi_document"
  },
  {
    "query": "What is the CEO's private password?",
    "expected_answer": ["not found", "unknown"],
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  },
  {
    "query": "What is the secret internal financial data?",
    "expected_answer": ["not found", "unknown"],
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  },
  {
    "query": "What is the personal bank account number of employees?",
    "expected_answer": ["not found", "unknown"],
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  },
  {
    "query": "Does the front end resume mention Node js? Answer Yes or No",
    "expected_answer": ["Yes"],
    "relevant_sources": ["Front End Resume.pdf"],
    "difficulty": "easy",
    "type": "fact"
  },
  {
    "query": "What year did Chaitanya complete his bachelor's degree?",
    "expected_answer": ["2023"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf"],
    "difficulty": "medium",
    "type": "fact"
  },
  {
    "query": "How does tata deal with their customers?",
    "expected_answer": ["processional", "fair", "transparent"],
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "Does the Tata Code emphasize unity among employees? Answer Yes or No",
    "expected_answer": ["Yes"],
    "relevant_sources": ["company policy Tata Code Of Conduct.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What is the purpose of normalization in DBMS?",
    "expected_answer": ["reduce redundancy", "reducing redundancy"],
    "relevant_sources": ["DBMS_notes.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "Does a primary key allow duplicate values? Answer Yes or No",
    "expected_answer": ["No"],
    "relevant_sources": ["DBMS_notes.pdf"],
    "difficulty": "medium",
    "type": "conceptual"
  },
  {
    "query": "What algorithm is described in the machine learning notes?",
    "expected_answer": ["K nearest neighbours", "KNN"],
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "easy",
    "type": "conceptual"
  },
  {
    "query": "What distance metric is commonly used in KNN?",
    "expected_answer": ["Euclidean distance"],
    "relevant_sources": ["Introduction to machine learning and K nearest neighbours_0.docx"],
    "difficulty": "hard",
    "type": "conceptual"
  },
  {
    "query": "Which employee has the highest salary in the employee dataset?",
    "expected_answer": ["Mark Newman"],
    "relevant_sources": ["Employee_data.csv"],
    "difficulty": "medium",
    "type": "numeric"
  },
  {
    "query": "Which department has the largest number of employees?",
    "expected_answer": ["Product"],
    "relevant_sources": ["employeeData.json"],
    "difficulty": "hard",
    "type": "numeric"
  },
  {
    "query": "How many departments exist in the employee dataset?",
    "expected_answer": ["5"],
    "relevant_sources": ["employeeData.csv"],
    "difficulty": "hard",
    "type": "numeric"
  },
  {
    "query": "What time is mentioned at the bottom of the society notice?",
    "expected_answer": ["10 AM"],
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "Who signed the society notice?",
    "expected_answer": ["Secretary"],
    "relevant_sources": ["society_notice.jpg"],
    "difficulty": "hard",
    "type": "ocr"
  },
  {
    "query": "What tax is applied after CGST on the receipt?",
    "expected_answer": ["SGST"],
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "medium",
    "type": "ocr"
  },
  {
    "query": "What is the total tax amount shown on the receipt?",
    "expected_answer": ["40.96", "41"],
    "relevant_sources": ["restaurant_receipt.jpg"],
    "difficulty": "hard",
    "type": "ocr"
  },
  {
    "query": "Which skills appear in both resumes?",
    "expected_answer": ["JavaScript"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf", "Front End Resume.pdf"],
    "difficulty": "hard",
    "type": "multi_document"
  },
  {
    "query": "Does both resumes mention React? Answer Yes or No",
    "expected_answer": ["Yes"],
    "relevant_sources": ["Chaitanya Shinde Resume updated.pdf", "Front End Resume.pdf"],
    "difficulty": "hard",
    "type": "multi_document"
  },
  {
    "query": "What confidential security clearance level is mentioned?",
    "expected_answer": ["not found", "unknown"],
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  },
  {
    "query": "What internal secret encryption key is listed?",
    "expected_answer": ["not found", "unknown"],
    "relevant_sources": [],
    "difficulty": "hard",
    "type": "negative"
  }
]


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def precision_at_k(retrieved, relevant, k):
    unique_retrieved = list(dict.fromkeys(retrieved))
    retrieved_k = unique_retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(retrieved_k) if retrieved_k else 0.0


def recall_at_k(retrieved, relevant, k):
    unique_retrieved = list(dict.fromkeys(retrieved))
    retrieved_k = unique_retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def f1_at_k(retrieved, relevant, k):
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def reciprocal_rank(retrieved, relevant):
    unique_retrieved = list(dict.fromkeys(retrieved))
    for i, r in enumerate(unique_retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    unique_retrieved = list(dict.fromkeys(retrieved))
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(unique_retrieved):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant)


# ============================================================================
# ANSWER NORMALIZATION
# ============================================================================

def normalize_answer(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.replace('.', ' ').replace(',', ' ').replace('%', ' ')
    text = text.replace('&', ' and ').replace('(', ' ').replace(')', ' ')
    text = text.replace('-', ' ').replace('/', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# ANSWER QUALITY METRICS
# ============================================================================

sim_model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(expected_answers, generated):
    generated_norm = normalize_answer(generated)
    if isinstance(expected_answers, str):
        expected_answers = [expected_answers]
    expected_norm = [normalize_answer(e) for e in expected_answers]
    all_texts = expected_norm + [generated_norm]
    embeddings = sim_model.encode(all_texts)
    gen_embedding = embeddings[-1]
    similarities = [
        cosine_similarity([embeddings[i]], [gen_embedding])[0][0]
        for i in range(len(expected_norm))
    ]
    return float(max(similarities)) if similarities else 0.0


def exact_match(expected_answers, generated):
    generated_norm = normalize_answer(generated)
    if isinstance(expected_answers, str):
        expected_answers = [expected_answers]
    for e in expected_answers:
        e_norm = normalize_answer(e)
        if e_norm in generated_norm or generated_norm in e_norm:
            return 1
    return 0


# ============================================================================
# LLM JUDGE FUNCTIONS  — use _call_judge_llm (NOT _call_llm_raw)
# ============================================================================

def _parse_judge_score(response: str, label: str) -> float:
    print(f"[{label} RAW] {response}")

    # --- Strip DeepSeek-R1 / thinking-model reasoning blocks ---
    # These models emit <think>...</think> before the actual answer.
    # Remove closed blocks first, then any unclosed opening tag and everything after.
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*",          "", cleaned,  flags=re.DOTALL).strip()

    # --- Find all decimal / integer candidates in [0, 1] ---
    # Prefer the LAST number found: models often reason "it is around 0.8"
    # and then conclude "Score: 0.8" — the last occurrence is the verdict.
    candidates = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", cleaned)
    if not candidates:
        # Broader fallback: grab any number and clamp it
        all_nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
        if all_nums:
            score = max(0.0, min(1.0, float(all_nums[-1])))
            print(f"[{label} PARSED-FALLBACK] {score}")
            return score
        print(f"[{label} ERROR] No number found — defaulting to 0.5")
        return 0.5

    score = max(0.0, min(1.0, float(candidates[-1])))
    print(f"[{label} PARSED] {score}")
    return score


def evaluate_faithfulness(answer, context, rag_system, judge_model):
    prompt = f"""Evaluate if the following answer is faithful to the provided context.
Faithfulness means the answer does not contradict the context and all claims are supported by it.

Context:
{context[:2000]}

Answer:
{answer}

Rate faithfulness 0-1:
- 1.0: All information supported by context
- 0.5: Mostly faithful, some unsupported claims
- 0.0: Contradicts context or major unsupported claims

Respond with ONLY a number between 0 and 1. No explanation. No text.
Example outputs: 0.0  0.5  1.0"""
    try:
        return _parse_judge_score(rag_system._call_judge_llm(prompt, judge_model), "FAITHFULNESS")
    except Exception as e:
        print(f"[FAITHFULNESS EXCEPTION] {e}")
        return 0.5


def evaluate_relevance(answer, query, rag_system, judge_model):
    prompt = f"""Evaluate if the following answer is relevant to the query.

Query: {query}

Answer: {answer}

Rate relevance 0-1:
- 1.0: Directly and fully addresses the query
- 0.5: Somewhat relevant, misses key aspects
- 0.0: Completely off-topic

Respond with ONLY a number between 0 and 1. No explanation. No text.
Example outputs: 0.0  0.5  1.0"""
    try:
        return _parse_judge_score(rag_system._call_judge_llm(prompt, judge_model), "RELEVANCE")
    except Exception as e:
        print(f"[RELEVANCE EXCEPTION] {e}")
        return 0.5


def evaluate_context_relevance(context, query, rag_system, judge_model):
    prompt = f"""Evaluate if the following context is relevant to the query.

Query: {query}

Context:
{context[:2000]}

Rate context relevance 0-1:
- 1.0: Contains information directly needed to answer the query
- 0.5: Somewhat relevant, tangential information
- 0.0: Completely irrelevant

Respond with ONLY a number between 0 and 1. No explanation. No text.
Example outputs: 0.0  0.5  1.0"""
    try:
        return _parse_judge_score(rag_system._call_judge_llm(prompt, judge_model), "CONTEXT_RELEVANCE")
    except Exception as e:
        print(f"[CONTEXT_RELEVANCE EXCEPTION] {e}")
        return 0.5


# ============================================================================
# STEP 1 — RETRIEVAL METRICS (model-independent, computed once)
# ============================================================================

def compute_retrieval_metrics(rag, test_set, top_k, user_id):
    print(f"\n{'='*70}")
    print("COMPUTING RETRIEVAL METRICS (Model-Independent)")
    print(f"{'='*70}")

    per_query_retrieval = {}

    for i, sample in enumerate(test_set):
        qid = i + 1
        query = sample["query"]
        relevant = [os.path.basename(r) for r in sample["relevant_sources"]]

        print(f"\n--- Query {qid}/{len(test_set)}: {query[:60]}... ---")

        result = rag.answer(
            question=query,
            google_id=user_id,
            top_k=top_k,
            model="llama3.2:1b-instruct-q4_K_M"
        )

        retrieved_sources = [
            os.path.basename(s["filename"])
            for s in result.get("sources", [])
        ]

        p     = precision_at_k(retrieved_sources, relevant, top_k)
        r     = recall_at_k(retrieved_sources, relevant, top_k)
        f1    = f1_at_k(retrieved_sources, relevant, top_k)
        rr    = reciprocal_rank(retrieved_sources, relevant)
        map_s = mean_average_precision(retrieved_sources, relevant)
        unique = list(dict.fromkeys(retrieved_sources))
        hits  = sum(1 for s in unique if s in relevant)

        per_query_retrieval[qid] = {
            "retrieved_sources": "|".join(retrieved_sources),
            "relevant_sources":  "|".join(relevant),
            "num_retrieved":     len(unique),
            "num_relevant":      len(relevant),
            "hits":              hits,
            "precision_at_k":   round(p,     4),
            "recall_at_k":      round(r,     4),
            "f1_at_k":          round(f1,    4),
            "mrr":              round(rr,    4),
            "map":              round(map_s, 4),
        }

        print(f"  Retrieved: {retrieved_sources}")
        print(f"  Relevant:  {relevant}")
        print(f"  P@{top_k}={p:.3f}  R@{top_k}={r:.3f}  F1={f1:.3f}  MRR={rr:.3f}  MAP={map_s:.3f}")

    return per_query_retrieval


# ============================================================================
# STEP 2 — PER-MODEL ANSWER QUALITY METRICS
# ============================================================================

def evaluate_model(rag, model_name, test_set, top_k):
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'='*70}")

    per_query_model = {}

    for i, sample in enumerate(test_set):
        qid = i + 1
        query = sample["query"]
        expected = sample["expected_answer"]

        print(f"\n--- Query {qid}/{len(test_set)}: {query[:60]}... ---")

        start = time.time()
        result = rag.answer(
            question=query,
            google_id=USER_ID,
            top_k=top_k,
            model=model_name
        )
        latency = time.time() - start

        generated      = result.get("answer", "")
        context_chars  = result.get("eval", {}).get("context_chars", 0)
        retrieved_count = result.get("retrieved_count", 0)
        intent         = result.get("eval", {}).get("intent", "")

        context = "\n\n".join([
            f"[From {s['filename']}]\n{s.get('text', '')}"
            for s in result.get("sources", [])
        ]) if result.get("sources") else ""

        sem_sim = semantic_similarity(expected, generated)
        e_match = exact_match(expected, generated)
        
        if context and generated:
            faith    = evaluate_faithfulness(generated, context, rag, judge_model=model_name)
            relev    = evaluate_relevance(generated, query, rag, judge_model=model_name)
            ctx_relv = evaluate_context_relevance(context, query, rag, judge_model=model_name)
        else:
            faith = relev = ctx_relv = 0.5

        expected_display = expected if isinstance(expected, str) else " | ".join(expected)
        print(f"  Expected:  {expected_display[:60]}")
        print(f"  Generated: {generated[:60]}")
        print(f"  SemanticSim={sem_sim:.3f}  ExactMatch={e_match}  "
              f"Faith={faith:.3f}  Relev={relev:.3f}  CtxRelv={ctx_relv:.3f}  "
              f"Latency={latency:.2f}s")

        per_query_model[qid] = {
            "generated_answer":    generated[:500],
            "intent_classified":   intent,
            "context_chars":       context_chars,
            "retrieved_count":     retrieved_count,
            "latency_s":           round(latency, 3),
            "semantic_similarity": round(sem_sim, 4),
            "exact_match":         e_match,
            "faithfulness":        round(faith,    4),
            "relevance":           round(relev,    4),
            "context_relevance":   round(ctx_relv, 4),
        }

    return per_query_model


# ============================================================================
# CSV WRITER — one row per query, all metrics as columns
#
# Column layout (52 rows × total columns):
#
#   QUERY METADATA (5 columns)
#     query_id, query, query_type, difficulty, expected_answer
#
#   RETRIEVAL METRICS — model-independent (10 columns)
#     retrieved_sources, relevant_sources,
#     num_retrieved, num_relevant, hits,
#     precision_at_K, recall_at_K, f1_at_K, mrr, map
#
#   PER MODEL — repeated for each model (10 columns × n_models)
#     {model}__generated_answer
#     {model}__intent_classified
#     {model}__context_chars
#     {model}__retrieved_count
#     {model}__latency_s
#     {model}__semantic_similarity
#     {model}__exact_match
#     {model}__faithfulness
#     {model}__relevance
#     {model}__context_relevance
# ============================================================================

def write_csv(test_set, per_query_retrieval, per_model_results, models, top_k, filepath):

    def safe_prefix(name):
        return re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")

    model_prefixes = [safe_prefix(m) for m in models]

    model_metric_keys = [
        "generated_answer",
        "intent_classified",
        "context_chars",
        "retrieved_count",
        "latency_s",
        "semantic_similarity",
        "exact_match",
        "faithfulness",
        "relevance",
        "context_relevance",
    ]

    header = [
        # Query metadata
        "query_id",
        "query",
        "query_type",
        "difficulty",
        "expected_answer",
        # Retrieval metrics (model-independent)
        "retrieved_sources",
        "relevant_sources",
        "num_retrieved",
        "num_relevant",
        "hits",
        f"precision_at_{top_k}",
        f"recall_at_{top_k}",
        f"f1_at_{top_k}",
        "mrr",
        "map",
    ]
    # Per-model columns
    for prefix in model_prefixes:
        for mk in model_metric_keys:
            header.append(f"{prefix}__{mk}")

    rows = []
    for i, sample in enumerate(test_set):
        qid = i + 1
        ret = per_query_retrieval.get(qid, {})
        expected_display = (
            " | ".join(sample["expected_answer"])
            if isinstance(sample["expected_answer"], list)
            else sample["expected_answer"]
        )

        row = {
            "query_id":       qid,
            "query":          sample["query"],
            "query_type":     sample.get("type", ""),
            "difficulty":     sample.get("difficulty", ""),
            "expected_answer": expected_display,
            "retrieved_sources": ret.get("retrieved_sources", ""),
            "relevant_sources":  ret.get("relevant_sources", ""),
            "num_retrieved":     ret.get("num_retrieved", ""),
            "num_relevant":      ret.get("num_relevant", ""),
            "hits":              ret.get("hits", ""),
            f"precision_at_{top_k}": ret.get("precision_at_k", ""),
            f"recall_at_{top_k}":    ret.get("recall_at_k", ""),
            f"f1_at_{top_k}":        ret.get("f1_at_k", ""),
            "mrr":               ret.get("mrr", ""),
            "map":               ret.get("map", ""),
        }

        for model_name, prefix in zip(models, model_prefixes):
            model_data = per_model_results.get(model_name, {}).get(qid, {})
            for mk in model_metric_keys:
                row[f"{prefix}__{mk}"] = model_data.get(mk, "")

        rows.append(row)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ CSV saved → {filepath}  ({len(rows)} rows × {len(header)} columns)")
    return header


# ============================================================================
# AGGREGATE SUMMARY
# ============================================================================

def _mean_std(values):
    """Return (mean, std) tuple, both rounded to 4dp."""
    import math
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return round(mean, 4), round(math.sqrt(variance), 4)


def print_summary(test_set, per_query_retrieval, per_model_results, models, top_k):
    n = len(test_set)
    ret_vals = list(per_query_retrieval.values())

    print(f"\n{'='*70}")
    print("AVERAGE RETRIEVAL METRICS (Model-Independent)  [mean ± std]")
    print(f"{'='*70}")
    for col, label in [
        ("precision_at_k", f"Precision@{top_k}"),
        ("recall_at_k",    f"Recall@{top_k}"),
        ("f1_at_k",        f"F1@{top_k}"),
        ("mrr",            "MRR"),
        ("map",            "MAP"),
    ]:
        vals = [r[col] for r in ret_vals]
        mean, std = _mean_std(vals)
        print(f"  {label:<20}: {mean} ± {std}")

    for model_name in models:
        model_data = per_model_results.get(model_name, {})
        vals = list(model_data.values())
        print(f"\n{'='*70}")
        print(f"AVERAGE METRICS — {model_name.upper()}  [mean ± std]")
        print(f"{'='*70}")
        for col in [
            "semantic_similarity", "exact_match",
            "faithfulness", "relevance", "context_relevance",
            "latency_s", "context_chars", "retrieved_count",
        ]:
            nums = [r[col] for r in vals if isinstance(r.get(col), (int, float))]
            mean, std = _mean_std(nums)
            if mean is not None:
                print(f"  {col:<25}: {mean} ± {std}")
            else:
                print(f"  {col:<25}: N/A")

    # ── Per query-type breakdown ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER QUERY-TYPE BREAKDOWN  (Exact Match & Semantic Similarity)")
    print(f"{'='*70}")
    type_indices: dict = {}
    for i, s in enumerate(test_set):
        qtype = s.get("type", "unknown")
        type_indices.setdefault(qtype, []).append(i + 1)

    for model_name in models:
        model_data = per_model_results.get(model_name, {})
        print(f"\n  Model: {model_name}")
        print(f"  {'Type':<20} {'N':>4}  {'ExactMatch':>12}  {'SemSim':>12}")
        print(f"  {'-'*55}")
        for qtype, qids in sorted(type_indices.items()):
            em_vals  = [model_data[qid]["exact_match"]         for qid in qids if qid in model_data]
            ss_vals  = [model_data[qid]["semantic_similarity"] for qid in qids if qid in model_data]
            em_mean, em_std = _mean_std(em_vals)
            ss_mean, ss_std = _mean_std(ss_vals)
            em_str = f"{em_mean}±{em_std}" if em_mean is not None else "N/A"
            ss_str = f"{ss_mean}±{ss_std}" if ss_mean is not None else "N/A"
            print(f"  {qtype:<20} {len(qids):>4}  {em_str:>12}  {ss_str:>12}")

    # ── Per difficulty breakdown ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER DIFFICULTY BREAKDOWN  (Exact Match & Semantic Similarity)")
    print(f"{'='*70}")
    diff_indices: dict = {}
    for i, s in enumerate(test_set):
        diff = s.get("difficulty", "unknown")
        diff_indices.setdefault(diff, []).append(i + 1)

    for model_name in models:
        model_data = per_model_results.get(model_name, {})
        print(f"\n  Model: {model_name}")
        print(f"  {'Difficulty':<12} {'N':>4}  {'ExactMatch':>12}  {'SemSim':>12}")
        print(f"  {'-'*45}")
        for diff in ["easy", "medium", "hard"]:
            qids = diff_indices.get(diff, [])
            if not qids:
                continue
            em_vals = [model_data[qid]["exact_match"]         for qid in qids if qid in model_data]
            ss_vals = [model_data[qid]["semantic_similarity"] for qid in qids if qid in model_data]
            em_mean, em_std = _mean_std(em_vals)
            ss_mean, ss_std = _mean_std(ss_vals)
            em_str = f"{em_mean}±{em_std}" if em_mean is not None else "N/A"
            ss_str = f"{ss_mean}±{ss_std}" if ss_mean is not None else "N/A"
            print(f"  {diff:<12} {len(qids):>4}  {em_str:>12}  {ss_str:>12}")


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
    return obj


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n--- RAG System Evaluation ---\n")

    embed = EmbeddingModel(os.getenv("EMBEDDING_MODEL"))
    CHROMA_DIR = os.path.abspath(os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))
    db = ChromaClient({"persist_directory": CHROMA_DIR})
    db.ensure_collection()

    collection = db.get_user_collection(USER_ID)
    print(f"Vector count: {collection.count()}")
    rag = RAGSystem(embed_model=embed, db_client=db)

    # ------------------------------------------------------------------
    # STEP 1: Retrieval metrics (model-independent, computed once)
    # ------------------------------------------------------------------
    per_query_retrieval = compute_retrieval_metrics(rag, TEST_SET, TOP_K, USER_ID)

    # ------------------------------------------------------------------
    # STEP 2: Answer quality per model
    # ------------------------------------------------------------------
    per_model_results  = {}

    for model in MODELS_TO_EVALUATE:
        try:
            model_data = evaluate_model(rag, model, TEST_SET, TOP_K)
            per_model_results[model] = model_data
        except Exception as e:
            print(f"[ERROR] evaluating {model}: {e}")
            per_model_results[model] = {}

    # ------------------------------------------------------------------
    # STEP 3: Write per-question CSV
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"evaluation_per_question_{timestamp}.csv"
    header    = write_csv(
        TEST_SET, per_query_retrieval, per_model_results,
        MODELS_TO_EVALUATE, TOP_K, csv_path
    )

    # ------------------------------------------------------------------
    # STEP 4: Print + save summary
    # ------------------------------------------------------------------
    print_summary(TEST_SET, per_query_retrieval, per_model_results, MODELS_TO_EVALUATE, TOP_K)

    n = len(TEST_SET)
    ret_vals = list(per_query_retrieval.values())

    # ── Retrieval averages + std ──────────────────────────────────────────
    avg_retrieval = {}
    for col in ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "map"]:
        vals = [r[col] for r in ret_vals]
        mean, std = _mean_std(vals)
        avg_retrieval[col] = {"mean": mean, "std": std}

    # ── Model averages + std ──────────────────────────────────────────────
    avg_model_metrics_full = {}
    for model in MODELS_TO_EVALUATE:
        model_data = per_model_results.get(model, {})
        vals = list(model_data.values())
        model_stats = {}
        for mk in ["semantic_similarity", "exact_match", "faithfulness",
                   "relevance", "context_relevance", "latency_s",
                   "context_chars", "retrieved_count"]:
            nums = [r[mk] for r in vals if isinstance(r.get(mk), (int, float))]
            mean, std = _mean_std(nums)
            model_stats[mk] = {"mean": mean, "std": std}
        avg_model_metrics_full[model] = model_stats

    # Keep the flat mean-only dict for backwards compatibility
    avg_model_metrics = {
        model: {mk: stats["mean"] for mk, stats in stats_dict.items()}
        for model, stats_dict in avg_model_metrics_full.items()
    }

    # ── Per query-type metrics ────────────────────────────────────────────
    type_indices: dict = {}
    for i, s in enumerate(TEST_SET):
        type_indices.setdefault(s.get("type", "unknown"), []).append(i + 1)

    per_type_metrics: dict = {}
    for qtype, qids in type_indices.items():
        per_type_metrics[qtype] = {}
        for model in MODELS_TO_EVALUATE:
            model_data = per_model_results.get(model, {})
            em_vals = [model_data[qid]["exact_match"]         for qid in qids if qid in model_data]
            ss_vals = [model_data[qid]["semantic_similarity"] for qid in qids if qid in model_data]
            em_mean, em_std = _mean_std(em_vals)
            ss_mean, ss_std = _mean_std(ss_vals)
            per_type_metrics[qtype][model] = {
                "n": len(qids),
                "exact_match":         {"mean": em_mean, "std": em_std},
                "semantic_similarity": {"mean": ss_mean, "std": ss_std},
            }

    # ── Per difficulty metrics ────────────────────────────────────────────
    diff_indices: dict = {}
    for i, s in enumerate(TEST_SET):
        diff_indices.setdefault(s.get("difficulty", "unknown"), []).append(i + 1)

    per_difficulty_metrics: dict = {}
    for diff, qids in diff_indices.items():
        per_difficulty_metrics[diff] = {}
        for model in MODELS_TO_EVALUATE:
            model_data = per_model_results.get(model, {})
            em_vals = [model_data[qid]["exact_match"]         for qid in qids if qid in model_data]
            ss_vals = [model_data[qid]["semantic_similarity"] for qid in qids if qid in model_data]
            em_mean, em_std = _mean_std(em_vals)
            ss_mean, ss_std = _mean_std(ss_vals)
            per_difficulty_metrics[diff][model] = {
                "n": len(qids),
                "exact_match":         {"mean": em_mean, "std": em_std},
                "semantic_similarity": {"mean": ss_mean, "std": ss_std},
            }

    summary = {
        "config": {
            "top_k":          TOP_K,
            "models":         MODELS_TO_EVALUATE,
            "test_set_size":  len(TEST_SET),
            "user_id":        USER_ID,
            "timestamp":      timestamp,
            "csv_columns":    header,
            "total_csv_cols": len(header),
        },
        "avg_retrieval_metrics":    avg_retrieval,
        "avg_model_metrics":        avg_model_metrics,          # flat means (backwards compat)
        "avg_model_metrics_with_std": avg_model_metrics_full,   # means + std
        "per_type_metrics":         per_type_metrics,
        "per_difficulty_metrics":   per_difficulty_metrics,
    }

    json_path = f"evaluation_summary_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(summary), f, indent=2)

    print(f"\n✓ Summary JSON  → {json_path}")
    print(f"✓ Per-question CSV → {csv_path}")
    print(f"\nAll {len(header)} CSV columns:")
    for col in header:
        print(f"  {col}")


if __name__ == "__main__":
    main()
