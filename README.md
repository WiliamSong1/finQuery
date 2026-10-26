# FinQuery

**Financial-document intelligence with hybrid retrieval, agentic analysis, and evidence verification.**

FinQuery lets users upload SEC annual and quarterly filings (10-Ks and 10-Qs), ask natural-language questions, and receive answers with references to the underlying documents. A Streamlit interface connects to a FastAPI backend that parses filings, indexes document passages, retrieves relevant evidence, and uses a LangGraph workflow to generate and verify answers.

> **Project status:** Working research prototype. Use public, non-sensitive filings for testing. The deployed backend currently uses a shared document collection and does not provide per-user document isolation or production-grade rate limiting.

## Features

- **Multi-document PDF ingestion:** Upload financial filings and index extracted text with document and page metadata.
- **Hybrid retrieval:** Combine Qdrant dense-vector search and BM25 keyword matching; rerank candidates using a cross-encoder.
- **Year-aware retrieval:** Scope retrieval to an appropriate filing year when the query and indexed document metadata permit it.
- **Agentic question answering:** Route research questions through a LangGraph workflow and generate grounded responses with Groq-hosted models.
- **Citation and claim verification:** Check cited passage IDs and validate quoted evidence against indexed source text before accepting answers.
- **Financial data utilities:** Includes SEC XBRL-based comparison functionality; the specialized deterministic comparison path is currently limited in scope and should not be assumed to support every company or metric.
- **Cloud deployment:** Dockerized FastAPI backend on AWS EC2, served behind Caddy HTTPS with API-key authentication; Streamlit frontend on Streamlit Community Cloud.

## Architecture

```text
Streamlit frontend (Streamlit Community Cloud)
        |
        | HTTPS + X-API-Key
        v
Caddy reverse proxy (AWS EC2)
        |
        v
FastAPI backend (Docker)
        |
        +-- PDF ingestion --> parsing / chunking / embeddings
        |                                      |
        |                                      v
        |                               Qdrant document index
        |
        +-- LangGraph research workflow
                 |
                 +-- dense retrieval (Qdrant)
                 +-- keyword retrieval (BM25)
                 +-- hybrid fusion + cross-encoder reranking
                 +-- optional year-scoped search
                 |
                 v
             Groq answer generation
                 |
                 v
          citations + claim verification
                 |
                 v
         answer and evidence references
```

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Language / API | Python, FastAPI |
| Frontend | Streamlit |
| Orchestration / inference | LangGraph, Groq |
| Retrieval | Qdrant, BM25, Sentence Transformers, cross-encoder reranking |
| Infrastructure | Docker, Docker Compose, AWS EC2, Caddy HTTPS |
| Evaluation | Custom Recall@5 / MRR@5 / nDCG@5 benchmarks, Ragas experiments |

## Retrieval Benchmark

A **20-question, manually labeled SEC filing benchmark** compared five retrieval approaches on the same indexed corpus. Reported times are **warm, sequential local measurements**, not end-to-end API latency.

| Method | Recall@5 | MRR@5 | nDCG@5 | Median latency |
| --- | ---: | ---: | ---: | ---: |
| Dense | 60% | 0.277 | 0.358 | 6.11 ms |
| BM25 | 45% | 0.283 | 0.323 | 2.21 ms |
| Hybrid | 60% | 0.353 | 0.415 | 10.10 ms |
| Reranked hybrid | 75% | 0.354 | 0.451 | 85.20 ms |
| **Year-aware reranked** | **85%** | **0.488** | **0.579** | **174.58 ms** |

**Result:** Year-aware reranked retrieval improved Recall@5 by **25 percentage points** over dense-only search (60% → 85%) on this dataset.

**Limitations:** This is a small, in-project benchmark, not an independent production evaluation. Evidence labels and corpus composition influence the scores; the additional retrieval accuracy also increases latency. Year-aware search falls back to general reranked retrieval for queries that do not specify exactly one year.

Evaluation scripts and labels are in [`evaluation/`](evaluation/).

## Repository Structure

```text
app/
  agents/              # LangGraph orchestration and query planning
  core/                # Configuration
  database/            # Qdrant storage
  generation/          # Answer generation and verification
  ingestion/           # PDF parsing and chunking
  retrieval/           # Dense, BM25, hybrid and reranked search
  tools/               # Financial data / calculation utilities
  main.py              # FastAPI app and routes
streamlit_demo/
  app.py               # Streamlit interface
evaluation/             # Retrieval and answer evaluation
Dockerfile
compose.yaml
requirements.txt
```

## Run Locally

### Prerequisites

- Python 3.13 (the project was developed and tested with Python 3.13)
- A Groq API key
- A descriptive SEC user agent with contact information for SEC requests

### 1. Clone and install

```bash
git clone https://github.com/WilliamSong1/finQuery.git
cd finQuery
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The GitHub URL above is the project's previously used repository URL. If the repository is private, GitHub access is required.

### 2. Configure environment variables

Create a local `.env` file in the repository root:

```dotenv
ENVIRONMENT=development
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
SEC_USER_AGENT=FinQuery Research contact@example.com
QDRANT_PATH=./data/qdrant
MAX_UPLOAD_MB=25
```

Never commit `.env`, `.streamlit/secrets.toml`, API keys, or SSH keys.

### 3. Start the backend

```bash
uvicorn app.main:app --reload --port 8000
```

- API health check: `http://localhost:8000/health`
- Interactive API docs: `http://localhost:8000/docs`

Upload public filings through `POST /documents` before asking document-based questions. A fresh local Qdrant index does **not** contain the filings from the EC2 deployment.

### 4. Start the frontend

Create `.streamlit/secrets.toml` in the repository root:

```toml
FINQUERY_API_URL = "http://localhost:8000"
FINQUERY_API_KEY = "local-development-key-if-required"
```

The development API can run without API-key enforcement depending on the configured environment. The production deployment requires a matching `FINQUERY_API_KEY` on the backend and in Streamlit secrets.

Start Streamlit in another terminal:

```bash
streamlit run streamlit_demo/app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

## API Overview

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Backend health check |
| `POST /documents` | Upload and index a PDF |
| `POST /search` | Retrieve relevant filing passages |
| `POST /ask` | Retrieval-backed question answering |
| `POST /agent/ask` | LangGraph research workflow |
| `POST /financial/compare` | Specialized structured financial comparison |

**Example agent request:**

```bash
curl -X POST http://localhost:8000/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What drove Apple\u0027s R&D expense growth in 2025?"}'
```

For the production backend, send an `X-API-Key` header over HTTPS. Do not expose port 8000 directly to the public internet.

## Docker / EC2 Deployment

The repository includes a `Dockerfile` and `compose.yaml` for the backend.

```bash
docker compose up --build -d
docker compose ps
docker compose logs --tail=50
```

The deployed configuration uses persistent Docker volumes for indexed Qdrant data and downloaded models. Configure the production `.env` on EC2, including `FINQUERY_API_KEY`, and recreate the container after environment changes:

```bash
docker compose up -d --force-recreate
```

The production layout is **Streamlit Cloud → HTTPS/Caddy → FastAPI on EC2 localhost:8000**. Keep SSH restricted to trusted IPs, expose only the reverse-proxy HTTPS ports, and store application secrets outside Git.

## Run the Retrieval Evaluation

Index the filings referenced by `evaluation/benchmark_questions.json` in your **local** Qdrant database. Then run:

```bash
python -m evaluation.validate_labels
python -m evaluation.retrieval_benchmark
```

The benchmark writes reports under `evaluation/results/`, including `retrieval_summary.csv`, `retrieval_by_category.csv`, and `retrieval_details.json`.

## Current Limitations and Next Steps

- **Shared index:** Documents uploaded by different sessions currently share one backend Qdrant collection. Do not use private or sensitive filings in a public demo.
- **Scope control:** Explicit per-user or per-document access controls and isolation are not yet implemented.
- **Accuracy:** Claim verification reduces unsupported statements but does not guarantee perfect financial or numerical correctness. Check financial conclusions against original filings.
- **Provider limits:** Groq request quotas and model availability may affect response reliability.
- **Specialized SEC calculations:** Deterministic financial analysis is not yet generalized to arbitrary companies and XBRL concepts.
- **Planned improvements:** User authentication, document-level scoping, source navigation, rate limits, broader financial metric support, and larger held-out evaluations.

## Disclaimer

FinQuery is a software engineering and financial-document research project. It is not investment advice; validate important figures and interpretations against original SEC filings.
