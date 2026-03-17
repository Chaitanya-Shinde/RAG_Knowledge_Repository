import os
import logging
from collections import defaultdict, Counter
from dotenv import load_dotenv
import time
import requests

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
- Do NOT invent information.
- Extract the correct information from the text.
- Give direct answers to the users query.
- Find the entire context for data
- If the user demands an explaination, only then focus on explaining the concept asked by the user.
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
        # Conversation Source Cache Retrieval
        # -------------------------

        recent_sources = self.get_recent_sources(google_id)
        print(f"\nrecent sources: {recent_sources}")
        if recent_sources:

            print(f"Checking recent sources: {recent_sources}")

            try:

                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where={"source": {"$in": recent_sources}}
                )

                docs = []
                best_similarity = 0

                if res and "documents" in res:

                    SIMILARITY_THRESHOLD = 0.10

                    for doc, meta, dist in zip(
                        res["documents"][0],
                        res["metadatas"][0],
                        res["distances"][0]
                    ):

                        similarity = 1 - (dist / 2)
                        best_similarity = max(best_similarity, similarity)
                        print(f"\nBest similarity: {best_similarity}")
                        print(f"Recent source similarity: {similarity}")

                        if similarity > SIMILARITY_THRESHOLD:

                            docs.append({
                                "text": doc,
                                "metadata": meta,
                                "score": similarity
                            })

                CACHE_CONFIDENCE_THRESHOLD = 0.25

                if docs and best_similarity > CACHE_CONFIDENCE_THRESHOLD:
                    print(f"Using conversation cache retrieval")
                    return docs

                print(f"Conversation cache similarity too low, falling back to global search")

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

        # # -------------------------
        # # DOCUMENT ROUTING
        # # -------------------------

        # best_source = None
        # best_similarity = 0

        # for source in sources:

        #     try:

        #         res = collection.query(
        #             query_embeddings=[q_emb],
        #             n_results=1,
        #             where={"source": source}
        #         )

        #         if res and res.get("distances"):

        #             dist = res["distances"][0][0]
        #             similarity = 1 - (dist / 2)

        #             if similarity > best_similarity:
        #                 best_similarity = similarity
        #                 best_source = source

        #     except Exception as e:
        #         logger.warning(f"Routing failed for {source}: {e}")

        # DOCUMENT_ROUTING_THRESHOLD = 0.30

        # if best_similarity < DOCUMENT_ROUTING_THRESHOLD:
        #     print(f"No document confidently matched. Best similarity {best_similarity}")
        #     return []
        # print(f"Selected document: {best_source}")
        # res = collection.query(
        #     query_embeddings=[q_emb],
        #     n_results=k,
        #     where={"source": best_source}
        # )
        # -------------------------
        # SIMPLE DOCUMENT NAME MATCH
        # -------------------------

        query_words = set(query.lower().split())

        for source in sources:

            name = source.lower().replace(".pdf","").replace(".csv","")
            name_words = set(name.split())

            overlap = query_words & name_words

            # if query shares words with filename
            if overlap:

                print(f"\nFilename match detected: {source}")

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

                        docs.append({
                            "text": doc,
                            "metadata": meta,
                            "score": similarity
                        })
                print(f"docs: {docs[:200]}")
                if docs:
                    return docs

        # -------------------------
        # GLOBAL VECTOR SEARCH
        # -------------------------

        res = collection.query(
            query_embeddings=[q_emb],
            n_results=k
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
                if similarity > 0.25:
                    docs.append({
                        "text": doc,
                        "metadata": meta,
                        "score": similarity
                    })
                print(f"Similarity: {similarity} | Source: {meta.get('source')}")

            RETRIEVAL_CONFIDENCE_THRESHOLD = 0.10
            if best_similarity < RETRIEVAL_CONFIDENCE_THRESHOLD:
                print(f"Low similarity retrieval: {best_similarity}, returning results anyway")
                

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
            "count"
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

            result = self.handle_dataset_query(question, google_id)

            if result:

                return {
                    "answer": result,
                    "sources": [],
                    "retrieved_count": 0
                }

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

    def handle_dataset_query(self, question, google_id):

        import pandas as pd
        from io import StringIO

        collection = self.db.get_user_collection(google_id)

        res = collection.get()

        if not res or "documents" not in res:
            return None

        # detect csv sources
        csv_sources = set(
            m["source"]
            for m in res["metadatas"]
            if m["source"].endswith(".csv")
        )

        if not csv_sources:
            return None

        # for now use the first dataset
        source = list(csv_sources)[0]

        res = collection.get(where={"source": source})

        text = "\n".join(res["documents"])

        try:

            df = pd.read_csv(StringIO(text))

            q = question.lower()

            if "max" in q or "highest" in q:

                col = df.select_dtypes(include="number").columns[-1]

                max_val = df[col].max()

                row = df[df[col] == max_val].iloc[0]

                return f"The highest {col} is {max_val}."

            if "min" in q or "lowest" in q:

                col = df.select_dtypes(include="number").columns[-1]

                min_val = df[col].min()

                return f"The lowest {col} is {min_val}."

            if "average" in q or "mean" in q:

                col = df.select_dtypes(include="number").columns[-1]

                avg = df[col].mean()

                return f"The average {col} is {round(avg,2)}."

        except Exception as e:

            logger.warning(f"Dataset analysis failed: {e}")

        return None