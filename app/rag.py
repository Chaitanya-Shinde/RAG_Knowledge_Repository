import os
import logging
from collections import defaultdict, Counter
from dotenv import load_dotenv
import time
import requests
from io import StringIO
import pandas as pd

import google.generativeai as genai

from .embeddings import EmbeddingModel
from .db_client import ChromaClient

load_dotenv()
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful AI assistant.

If the context is empty or irrelevant, answer normally like a chatbot.


Rules:
- You may have access to document context. If the context is useful, use it to answer the question.
- Use ONLY the information from the provided context.
- Do NOT explain your thinking.
- USE your reasoning.
- Extract the correct information from the text.
- Give direct answers to the users query.
- Find the entire context for data
- If the user demands an explanation, only then focus on explaining the concept asked by the user.
- Combine relevant information across chunks to produce a full response.


Context:
{context}

User Question:
{question}
"""


class RAGSystem:
    def __init__(self, embed_model: EmbeddingModel,  db_client: ChromaClient,model="llama3.2:1b"):
        self.embed_model = embed_model
        self.db = db_client
        self.db.ensure_collection()
        self.local_llm = model

        # -------------------------
        # Conversation memory
        # -------------------------

        # Stores recently used document sources per user
        # Example:
        # {
        #   "user123": ["esa_doc.pdf", "ml_notes.pdf"]
        # }
        self.recent_sources = {}

        # limit memory size
        self.max_recent_sources = 15
        
        self.api_key = os.getenv("GEMINI_API_KEY")

        if self.api_key: 
            genai.configure(api_key=self.api_key)

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # -------------------------
    # Conversation Source Cache
    # -------------------------

    def get_recent_sources(self, google_id):

        return self.recent_sources.get(google_id, [])


    def update_recent_sources(self, google_id, source):

        if google_id not in self.recent_sources:
            self.recent_sources[google_id] = []

        sources = self.recent_sources[google_id]

        # avoid duplicates
        if source not in sources:
            sources.append(source)

        # keep only last N
        self.recent_sources[google_id] = sources[-self.max_recent_sources:]
        
    # -------------------------
    # DOCUMENT INDEXING
    # -------------------------

    def index_documents(self, docs):
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]

        embeddings = self.embed_model.embed_documents(texts)

        self.db.add_documents(ids, texts, metas, embeddings.tolist())

        print(f"Indexed {len(docs)} documents")

        return {"indexed": len(docs)}

    # -------------------------
    # RETRIEVAL
    # -------------------------

    def retrieve(self, query, google_id, k=6):
        q_emb = self.embed_model.embed_query(query)

        collection = self.db.get_user_collection(google_id)

        # -------------------------
        # Recent source cache retrieval
        # -------------------------
        recent_sources = self.get_recent_sources(google_id)
        print(f"\nrecent sources: {recent_sources}")

        if recent_sources:
            try:
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where={"source": {"$in": recent_sources}}
                )

                docs = []
                best_similarity = 0

                if res and "documents" in res:
                    for doc, meta, dist in zip(
                        res["documents"][0],
                        res["metadatas"][0],
                        res["distances"][0]
                    ):
                        similarity = 1 - (dist / 2)
                        best_similarity = max(best_similarity, similarity)

                        if similarity >= 0.25:
                            docs.append({
                                "text": doc,
                                "metadata": meta,
                                "score": similarity
                            })

                if docs and best_similarity >= 0.30:
                    print("Using strong recent source cache retrieval")
                    return docs

            except Exception as e:
                logger.warning(f"Conversation cache retrieval failed: {e}")

        # Get all document sources
        res_all = collection.get()
        if not res_all or "metadatas" not in res_all:
            return []

        sources = list({
            m.get("source")
            for m in res_all["metadatas"]
            if m.get("source")
        })

        # -------------------------
        # Filename match route
        # -------------------------
        query_words = set(query.lower().split())
        for source in sources:
            name = source.lower().replace("_", " ").replace(".", " ")
            name_words = set(name.split())

            if query_words & name_words:
                print(f"Filename match detected: {source}")
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where={"source": source}
                )

                docs = []
                if res and "documents" in res:
                    for doc, meta, dist in zip(
                        res["documents"][0],
                        res["metadatas"][0],
                        res["distances"][0]
                    ):
                        similarity = 1 - (dist / 2)
                        if similarity >= 0.10:
                            docs.append({
                                "text": doc,
                                "metadata": meta,
                                "score": similarity
                            })

                if docs:
                    return docs

        # -------------------------
        # Global vector search + source-level reroute
        # -------------------------
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=100
        )

        if not res or "documents" not in res:
            return []

        hits = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        ):
            similarity = 1 - (dist / 2)
            if similarity >= 0.15 and meta.get("source"):
                hits.append({
                    "text": doc,
                    "metadata": meta,
                    "score": similarity
                })

        if not hits:
            return []

        source_best_score = {}
        for hit in hits:
            src = hit["metadata"]["source"]
            source_best_score[src] = max(source_best_score.get(src, 0), hit["score"])

        best_source = max(source_best_score, key=lambda s: source_best_score[s])
        best_source_score = source_best_score[best_source]

        print(f"Selected best source: {best_source} with score {best_source_score}")

        if best_source_score < 0.30:
            # If no source has strong confidence, return top hits from all sources
            return sorted(hits, key=lambda x: x["score"], reverse=True)[:k]

        res_source = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            where={"source": best_source}
        )

        docs = []
        if res_source and "documents" in res_source:
            for doc, meta, dist in zip(
                res_source["documents"][0],
                res_source["metadatas"][0],
                res_source["distances"][0]
            ):
                similarity = 1 - (dist / 2)
                if similarity >= 0.15:
                    docs.append({
                        "text": doc,
                        "metadata": meta,
                        "score": similarity
                    })

        return docs
    # -------------------------
    # CONTEXT EXPANSION
    # -------------------------

    def expand_context(self, docs, google_id, max_chunks_per_doc=3):

        collection = self.db.get_user_collection(google_id)

        sources = set(
            d["metadata"].get("source")
            for d in docs
        )

        expanded_docs = []

        for source in sources:

            try:

                res = collection.get(where={"source": source})

                if not res or "documents" not in res:
                    continue

                docs_for_source = list(
                    zip(res["documents"], res["metadatas"])
                )

                docs_for_source = sorted(
                    docs_for_source,
                    key=lambda x: x[1].get("chunk_index", 0)
                )

                docs_for_source = docs_for_source[:max_chunks_per_doc]

                for text, meta in docs_for_source:

                    text_lower = text.lower()

                    # remove low quality chunks
                    if "table of contents" in text_lower or "index" in text_lower:
                        continue

                    expanded_docs.append({
                        "text": text,
                        "metadata": meta
                    })

            except Exception as e:

                logger.warning(f"Context expansion failed for {source}: {e}")

        return expanded_docs

    # -------------------------
    # LLM CALL
    # -------------------------

    def call_gemini(self, question, context, max_tokens=1000):

        if not self.api_key:
            return "Gemini API key missing."

        model = genai.GenerativeModel(self.model_name)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # Extra instruction to force numeric display, not recipe text
        prompt += "\n\nIMPORTANT: When answered in dataset mode, return concrete numeric values only; do NOT return instructions, SQL templates, or step-by-step recipe text."

        retries = 3

        for attempt in range(retries):

            try:

                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens
                    )
                )

                return response.text

            except Exception as e:

                if "429" in str(e):

                    wait_time = 10
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                logger.error(f"Gemini API error: {str(e)}")
                return f"Error calling Gemini API: {str(e)}"

        return "LLM rate limit exceeded. Please try again later."

    def call_ollama(self, prompt, context, model="ollama", max_tokens=500):

        try:

            if model == "deepseek":
                ollama_model = "deepseek-r1:1.5b"
            else:
                ollama_model = "llama3.2:1b"

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "num_ctx": 1500,
                        "num_predict": 1000,
                        "temperature": 0.2
                    }
                },
                timeout=600
            )
            

            data = response.json()

            message = data.get("message", {})

            content = message.get("content", "").strip()
            thinking = message.get("thinking", "").strip()

            if not content and thinking:
                return thinking

            return content

        except Exception as e:

            logger.error(f"Ollama error: {str(e)}")

            return f"Ollama error: {str(e)}"
    
    # -------------------------
    # DATASET QUERY DETECTION
    # -------------------------

    def is_dataset_query(self, query):

        keywords = [
            "max", "maximum", "highest",
            "min", "minimum", "lowest",
            "average", "mean",
            "sum", "total",
            "count", "stats", "statistic", "statistics", "salary", "datasheet", "dataset"
        ]

        q = query.lower()

        return any(k in q for k in keywords)

    # -------------------------
    # MAIN RAG PIPELINE
    # -------------------------

    def answer(self, question, google_id, top_k, model="gemini"):

        if not question or len(question.strip()) == 0:
            return {
                "answer": "Question cannot be empty",
                "sources": [],
                "retrieved_count": 0
            }

        # -------------------------
        # DATASET QUERY HANDLER
        # -------------------------

        if self.is_dataset_query(question):

            print("Dataset query detected")

            llm_result, llm_source = \
                self.handle_dataset_query_via_llm(
                    question,
                    google_id,
                    model
                )

            if llm_result:

                return {
                    "answer": llm_result,
                    "sources": [{
                        "filename": llm_source or "unknown",
                        "snippet": ""
                    }],
                    "retrieved_count": 0
                }

            # fallback to normal RAG if dataset failed
            print("Dataset handler failed — falling back to retrieval")

        # -------------------------
        # RETRIEVE
        # -------------------------

        # -------------------------
        # INTENT GATE
        # -------------------------

        if self.is_smalltalk(question):

            print("Smalltalk detected → skipping retrieval")

            return {
                "answer": self.generate(question, "", model),
                "sources": [],
                "retrieved_count": 0
            }

        # Step 2: retrieval
        retrieved_docs = self.retrieve(
            question,
            google_id,
            k=top_k
        )

        # USE_CONTEXT_THRESHOLD = 0.10

        # docs = [
        #     d for d in retrieved_docs
        #     if d["score"] >= USE_CONTEXT_THRESHOLD
        # ]

        print("\nretrieved_docs ",retrieved_docs)

        if not retrieved_docs:

            # fallback to normal chatbot
            llm_out = self.generate(question, "", model)

            return {
                "answer": llm_out,
                "sources": [],
                "retrieved_count": 0
            }

        # -------------------------
        # SELECT SINGLE DOCUMENT
        # -------------------------

        source_counts = Counter(
            d["metadata"].get("source", "unknown")
            for d in retrieved_docs
        )

        best_source = source_counts.most_common(1)[0][0]
        # update conversation memory
        self.update_recent_sources(google_id, best_source)

        docs = [
            d for d in retrieved_docs
            if d["metadata"].get("source") == best_source
        ]

        # -------------------------
        # SORT CHUNKS
        # -------------------------
        

        docs = sorted(
            docs,
            key=lambda x: x["metadata"].get("chunk_index", 0)
        )


        # -------------------------
        # BUILD CONTEXT (ONLY ONE DOC)
        # -------------------------

        if docs:
            context = "\n\n".join(d["text"] for d in docs)
        else:
            context = ""
        # context = "\n\n".join(d["text"] for d in docs)

        context = f"Document: {best_source}\n\n{context}"

        # -------------------------
        # GENERATE ANSWER
        # -------------------------
        print("\nChunks sent:", len(docs))
        print("\nContext chars:", len(context))

        
        llm_out = self.generate(question, context, model)

        # -------------------------
        # SOURCES
        # -------------------------

        sources = [{
            "filename": best_source,
            "snippet": docs[0]["text"][:200] if docs else ""
        }]

        return {
            "answer": llm_out,
            "sources": sources,
            "retrieved_count": len(docs)
        }
    

    def generate(self, question, context, model="gemini"):

        if context:
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
        else:
            prompt = question
    
        if model == "ollama":
            return self.call_ollama(prompt, context, model='ollama')

        if model == "gemini":
            return self.call_gemini(prompt, context)
        
        if model == "deepseek":
            return self.call_ollama(prompt, context, model='deepseek')


        return "Unknown model selected"

    
    def find_best_csv_data(self, question, google_id):

        from io import StringIO
        import pandas as pd
        import re

        collection = self.db.get_user_collection(google_id)

        res_all = collection.get()

        if not res_all or "metadatas" not in res_all:
            return None, None

        # Find CSV sources
        csv_sources = [
            m["source"]
            for m in res_all["metadatas"]
            if m.get("source", "").lower().endswith(".csv")
        ]

        if not csv_sources:
            return None, None

        q_lower = question.lower()

        best_source = None

        # Try filename match
        for src in csv_sources:

            if any(word in src.lower() for word in q_lower.split()):
                best_source = src
                break

        if not best_source:
            best_source = csv_sources[0]

        try:

            # Load stored documents
            source_data = collection.get(
                where={"source": best_source}
            )

            if not source_data or "documents" not in source_data:
                return None, best_source

            raw_text = "\n".join(
                source_data["documents"]
            )

            print("\nRaw text preview:\n", raw_text[:500])

            # -------------------------
            # Try normal CSV first
            # -------------------------

            try:

                df = pd.read_csv(
                    StringIO(raw_text),
                    low_memory=False
                )

                if len(df.columns) == 1:

                    raise ValueError(
                        "Single-column detected"
                    )

            except Exception:

                print("Trying table reconstruction...")

                # -------------------------
                # Rebuild table from spaces
                # -------------------------

                lines = raw_text.split("\n")

                cleaned_lines = []

                for line in lines:

                    line = line.strip()

                    if not line:
                        continue

                    # Replace multiple spaces with comma
                    line = re.sub(
                        r"\s{2,}",
                        ",",
                        line
                    )

                    cleaned_lines.append(line)

                rebuilt_text = "\n".join(cleaned_lines)

                df = pd.read_csv(
                    StringIO(rebuilt_text)
                )

            # -------------------------
            # Convert numeric columns
            # -------------------------

            for col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="ignore"
                )

            print("\nDetected columns:", df.columns.tolist())
            print("\nFirst rows:\n", df.head())

            return df, best_source

        except Exception as e:

            logger.warning(
                f"CSV reconstruction failed: {e}"
            )

            return None, best_source

    def profile_dataframe(self, df):

        import numpy as np

        profile = {}

        profile["row_count"] = len(df)
        profile["column_count"] = len(df.columns)

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        profile["numeric_columns"] = numeric_cols

        profile["columns"] = []

        for col in df.columns:

            col_data = df[col]

            col_info = {
                "name": col,
                "dtype": str(col_data.dtype),
                "missing": int(col_data.isna().sum()),
                "unique": int(col_data.nunique())
            }

            if col in numeric_cols:

                stats = col_data.describe()

                col_info["stats"] = {
                    "count": float(stats.get("count", 0)),
                    "min": float(stats.get("min", 0)),
                    "max": float(stats.get("max", 0)),
                    "mean": float(stats.get("mean", 0)),
                    "median": float(stats.get("50%", 0)),
                    "std": float(stats.get("std", 0))
                }

            profile["columns"].append(col_info)

        return profile

    def build_dataset_math_context(self, df, source):

        profile = self.profile_dataframe(df)

        sample_rows = df.head(8).to_csv(
            index=False
        )

        numeric_cols = profile["numeric_columns"]

        correlation_text = ""

        if len(numeric_cols) > 1:

            corr = df[numeric_cols].corr().round(3)

            correlation_text = "\nCORRELATIONS:\n"
            correlation_text += corr.to_string()
            correlation_text += "\n"

        context = f"""
    DATASET SOURCE: {source}

    DATASET OVERVIEW:
    Rows: {profile['row_count']}
    Columns: {profile['column_count']}

    NUMERIC COLUMNS:
    {numeric_cols}

    COLUMN DETAILS:
    """

        for col in profile["columns"]:

            context += f"""
    Column: {col['name']}
    Type: {col['dtype']}
    Missing: {col['missing']}
    Unique: {col['unique']}
    """

            if "stats" in col:

                s = col["stats"]

                context += f"""
    Statistics:
    Count: {s['count']}
    Min: {s['min']}
    Max: {s['max']}
    Mean: {s['mean']}
    Median: {s['median']}
    Std: {s['std']}
    """

        context += correlation_text

        context += f"""

    SAMPLE DATA:
    {sample_rows}

    INSTRUCTIONS:
    Answer using dataset facts.
    Return numeric answers when possible.
    """

        return context

    def handle_dataset_query_via_llm(
        self,
        question,
        google_id,
        model="gemini"
    ):

        df, source = self.find_best_csv_data(
            question,
            google_id
        )

        if df is None or df.empty:

            return None, source

        try:

            context = self.build_dataset_math_context(
                df,
                source
            )

            answer = self.generate(
                question,
                context,
                model
            )

            return answer, source

        except Exception as e:

            logger.error(
                f"Dataset math failed: {e}"
            )

            return None, source

    # -------------------------
    # INTENT DETECTION
    # -------------------------

    def is_smalltalk(self, query):

        smalltalk = [
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "good morning",
            "good evening",
            "how are you",
            "what's up",
            "whats up"
        ]

        q = query.lower().strip()

        return q in smalltalk
    
    # -------------------------
    # DATASET ANALYSIS
    # -------------------------

    # def handle_dataset_query(self, question, google_id):

    #     import numpy as np
    #     import pandas as pd
    #     import re
    #     from io import StringIO

    #     collection = self.db.get_user_collection(google_id)

    #     # first, pick the most likely CSV source from retrieval or filename hints
    #     csv_sources = []
    #     res_all = collection.get()
    #     if res_all and "metadatas" in res_all:
    #         csv_sources = [
    #             m["source"] for m in res_all["metadatas"]
    #             if m.get("source", "").lower().endswith(".csv")
    #         ]

    #     q_lower = question.lower()

    #     best_source = None
    #     if "salary" in q_lower and csv_sources:
    #         salary_csvs = [src for src in csv_sources if "salary" in src.lower()]
    #         best_source = salary_csvs[0] if salary_csvs else csv_sources[0]

    #     if not best_source:
    #         # choose best candidate by semantic retrieval score (including non-CSV fallback)
    #         retrieved_docs = self.retrieve(question, google_id, k=20)
    #         source_scores = {}
    #         for d in retrieved_docs:
    #             src = d["metadata"].get("source")
    #             score = d.get("score", 0)
    #             if src:
    #                 source_scores[src] = max(source_scores.get(src, 0), score)
    #         if source_scores:
    #             best_source = max(source_scores, key=source_scores.get)

    #     if not best_source and csv_sources:
    #         best_source = csv_sources[0]

    #     # Build strong CSV stats if possible
    #     if best_source and best_source.lower().endswith(".csv"):
    #         try:
    #             source_data = collection.get(where={"source": best_source})
    #             if source_data and "documents" in source_data:
    #                 raw_text = "\n".join(source_data["documents"])
    #                 df = pd.read_csv(StringIO(raw_text))

    #                 if not df.empty:
    #                     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    #                     if not numeric_cols:
    #                         for col in df.columns:
    #                             coerced = pd.to_numeric(df[col], errors="coerce")
    #                             if coerced.notna().sum() > 0:
    #                                 df[col] = coerced
    #                         numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    #                     if numeric_cols:
    #                         selected_col = None
    #                         if "salary" in q_lower:
    #                             salary_cols = [c for c in numeric_cols if "salary" in str(c).lower()]
    #                             if salary_cols:
    #                                 selected_col = salary_cols[0]

    #                         for col in numeric_cols:
    #                             if col.lower() in q_lower and selected_col is None:
    #                                 selected_col = col

    #                         if not selected_col:
    #                             selected_col = numeric_cols[0]

    #                         series = df[selected_col].dropna().astype(float)
    #                         stats_summary = series.describe().to_dict()

    #                         answers = []
    #                         if "max" in q_lower or "highest" in q_lower:
    #                             answers.append(f"Highest {selected_col} = {stats_summary['max']}")
    #                         if "min" in q_lower or "lowest" in q_lower:
    #                             answers.append(f"Lowest {selected_col} = {stats_summary['min']}")
    #                         if "average" in q_lower or "mean" in q_lower:
    #                             answers.append(f"Average {selected_col} = {round(stats_summary['mean'], 2)}")
    #                         if "sum" in q_lower or "total" in q_lower:
    #                             answers.append(f"Total {selected_col} = {round(series.sum(), 2)}")
    #                         if "count" in q_lower or "rows" in q_lower:
    #                             answers.append(f"Row count = {int(stats_summary['count'])}")

    #                         if "statistic" in q_lower or "statistics" in q_lower or "describe" in q_lower or "summary" in q_lower:
    #                             answers.append(
    #                                 f"{selected_col} statistics (count={int(stats_summary['count'])}, mean={round(stats_summary['mean'],2)}, "
    #                                 f"std={round(stats_summary['std'],2)}, min={stats_summary['min']}, 25%={stats_summary['25%']}, "
    #                                 f"50%={stats_summary['50%']}, 75%={stats_summary['75%']}, max={stats_summary['max']})"
    #                             )

    #                         if not answers:
    #                             answers.append(
    #                                 f"Dataset {best_source} column {selected_col} stats: count={int(stats_summary['count'])}, "
    #                                 f"mean={round(stats_summary['mean'],2)}, min={stats_summary['min']}, max={stats_summary['max']}"
    #                             )

    #                         sample_data = series.head(5).tolist()
    #                         answers.append(f"Sample values: {sample_data}")
    #                         result = "; ".join(answers) + f" (source: {best_source})"
    #                         return result

    #         except Exception as e:
    #             logger.warning(f"Dataset CSV analysis failed: {e}")

    #     # Non-CSV fallback: extract relevant numeric values from retrieved docs
    #     retrieved_docs = self.retrieve(question, google_id, k=20)
    #     if not retrieved_docs:
    #         return None

    #     text = "\n".join(d.get("text", "") for d in retrieved_docs)
    #     if not text.strip():
    #         return None

    #     numbers = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    #     values = []
    #     for n in numbers:
    #         norm = n.replace(",", "")
    #         try:
    #             values.append(float(norm))
    #         except ValueError:
    #             continue

    #     if not values:
    #         return None

    #     arr = np.array(values, dtype=float)
    #     max_v = float(arr.max())
    #     min_v = float(arr.min())
    #     mean_v = float(arr.mean())
    #     total_v = float(arr.sum())
    #     count_v = int(arr.size)

    #     answers = []
    #     if "max" in q_lower or "highest" in q_lower:
    #         answers.append(f"Highest value found = {max_v}")
    #     if "min" in q_lower or "lowest" in q_lower:
    #         answers.append(f"Lowest value found = {min_v}")
    #     if "average" in q_lower or "mean" in q_lower:
    #         answers.append(f"Average value found = {round(mean_v,2)}")
    #     if "sum" in q_lower or "total" in q_lower:
    #         answers.append(f"Total value found = {round(total_v,2)}")
    #     if "count" in q_lower or "rows" in q_lower:
    #         answers.append(f"Numeric count = {count_v}")

    #     if "statistic" in q_lower or "statistics" in q_lower or "describe" in q_lower or "summary" in q_lower:
    #         answers.append(f"Fallback stats from text extraction: count={count_v}, mean={round(mean_v,2)}, min={min_v}, max={max_v}, std={round(float(np.std(arr)),2)}")

    #     if not answers:
    #         answers.append(f"Text-value stats: count={count_v}, mean={round(mean_v,2)}, min={min_v}, max={max_v}")

    #     result = "; ".join(answers) + f" (source: {best_source or 'retrieved documents'})"
    #     return result
