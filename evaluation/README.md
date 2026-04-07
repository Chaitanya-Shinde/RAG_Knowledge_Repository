# RAG System Evaluation for Research Paper

This evaluation framework provides comprehensive metrics to assess the performance of Retrieval-Augmented Generation (RAG) systems, specifically designed for demonstrating LLM implementation in knowledge repository systems.

## Evaluation Techniques Implemented

### 1. Retrieval Metrics
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that are retrieved
- **F1@K**: Harmonic mean of precision and recall
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant document
- **Mean Average Precision (MAP)**: Average precision across all queries

### 2. Answer Quality Metrics
- **Semantic Similarity**: Cosine similarity between expected and generated answers using Sentence-BERT
- **Exact Match**: Binary metric for exact answer matching (case-insensitive)

### 3. LLM-Judged Metrics
- **Faithfulness**: Does the answer contradict or hallucinate beyond the provided context?
- **Relevance**: How well does the answer address the query?
- **Context Relevance**: How relevant is the retrieved context to the query?

### 4. Performance Metrics
- **Latency**: Response time for intent classification, retrieval, and generation

## Models Evaluated
- **Gemini 2.5 Flash**: Large frontier model with detailed prompts
- **Llama 3.2 1B**: Small local model with simplified prompts
- **DeepSeek-R1 1.5B**: Reasoning-focused local model

## Test Set
The evaluation uses a curated test set with:
- Natural language queries about user documents
- Expected relevant source documents
- Ground truth answers for comparison
- Context relevance criteria

## Usage

```bash
cd /path/to/project
python evaluation/evaluate.py
```

Results are saved to `evaluation_results.json` and displayed in the console.

## Research Paper Applications

### Key Findings to Demonstrate:
1. **Retrieval Effectiveness**: Compare precision/recall across models to show embedding quality
2. **Model-Specific Performance**: Show how prompt complexity affects different model sizes
3. **Faithfulness Trade-offs**: Analyze how smaller models balance speed vs. hallucination
4. **Enterprise Readiness**: Evaluate latency and accuracy for production deployment

### Additional Evaluation Techniques to Consider:

#### Human Evaluation
- **Answer Correctness**: Rate answers on a 1-5 scale for factual accuracy
- **Answer Fluency**: Assess grammatical quality and coherence
- **Source Attribution**: Verify if answers correctly cite sources

#### Advanced Metrics
- **BLEU/ROUGE**: N-gram overlap for text similarity
- **BERTScore**: Transformer-based semantic similarity
- **RAGAS**: Comprehensive RAG evaluation framework

#### Ablation Studies
- **No Retrieval**: Compare RAG vs. direct LLM answering
- **Different K Values**: Analyze retrieval quality vs. context length
- **Embedding Models**: Compare different embedding strategies

#### Robustness Testing
- **Adversarial Queries**: Test with misleading or ambiguous questions
- **Out-of-Domain**: Evaluate on topics not in the knowledge base
- **Multi-turn Conversations**: Assess context retention over chat history

## Interpreting Results

### Retrieval Metrics
- High precision indicates good relevance filtering
- High recall shows comprehensive document coverage
- F1 balances both aspects for overall retrieval quality

### Answer Quality
- Semantic similarity measures answer correctness
- LLM-judged metrics provide nuanced quality assessment
- Faithfulness is critical for enterprise knowledge systems

### Model Comparison
- Larger models (Gemini) typically show better quality but higher latency
- Smaller models (Llama, DeepSeek) offer faster responses with acceptable quality
- Choose based on enterprise requirements for speed vs. accuracy

## Future Enhancements

1. **Larger Test Sets**: Expand to hundreds of queries for statistical significance
2. **Domain-Specific Evaluation**: Tailor metrics to specific knowledge domains
3. **Multi-Modal Evaluation**: Include image/document understanding
4. **Real-Time Monitoring**: Continuous evaluation in production
5. **User Feedback Integration**: Incorporate human ratings into the system

This evaluation framework demonstrates the practical implementation of RAG systems and provides empirical evidence for model selection in enterprise knowledge repository applications.</content>
<parameter name="filePath">g:\Applications\Docs\Msc College notes\Semester 4\RAGKR\evaluation\README.md