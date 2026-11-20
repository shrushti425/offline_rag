# EdgeRAG: A Privacy-Preserving Offline Retrieval-Augmented Generation System for NVIDIA Jetson Nano

A document question-answering system that runs entirely on-device. Documents are indexed, retrieved, and answered locally on a 4 GB NVIDIA Jetson Nano, with no cloud services, no API keys, and no network access required at inference time.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Project Objectives](#project-objectives)
4. [Problem Statement](#problem-statement)
5. [Key Features](#key-features)
6. [System Architecture](#system-architecture)
7. [Tech Stack](#tech-stack)
8. [Repository Structure](#repository-structure)
9. [Installation and Usage](#installation-and-usage)
10. [Deployment on NVIDIA Jetson Nano](#deployment-on-nvidia-jetson-nano)
11. [Evaluation](#evaluation)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
14. [Acknowledgements](#acknowledgements)

---

## Overview

EdgeRAG lets a user upload PDF documents and query them in natural language. The system retrieves the most relevant passages from the uploaded documents and passes only those passages to a locally hosted language model, which generates an answer with page-level source citations.

The system was developed on macOS (Ollama + TinyLlama) and deployed on an NVIDIA Jetson Nano (llama.cpp with CUDA, 4-bit quantised TinyLlama). Only the LLM wrapper differs between the two environments; the ingestion, embedding, retrieval, and interface layers are shared.

## Motivation

Large language models are normally run on cloud infrastructure or high-end GPUs, and most retrieval-augmented generation (RAG) systems assume the same. This project began as an effort to learn what it takes to deploy a language-model application on small, resource-constrained hardware, and to find out where such a deployment actually breaks down.

The NVIDIA Jetson Nano (4 GB RAM, roughly 99 USD, low power draw) was chosen as the target because its limits are severe enough to force real engineering decisions about model size, quantisation, memory use, and latency. The same limits match real deployments: facilities with restricted networks, organisations handling confidential records, and settings where API costs or connectivity are barriers.

## Project Objectives

1. Determine whether a complete RAG pipeline (ingestion, embedding, retrieval, and generation) can run on a 4 GB edge device without any cloud dependency.
2. Understand the practical constraints of resource-limited deployment: memory ceiling, inference latency, power draw, and the effect of 4-bit quantisation on answer quality.
3. Learn the deployment workflow for embedded hardware, including building llama.cpp with CUDA on an ARM64 device that Ollama does not support.
4. Design the system so that development and deployment share one codebase, with only the LLM module changing between macOS and the Jetson Nano.
5. Measure and report performance honestly, so the results show both what works and what does not.

## Problem Statement

Cloud-hosted language models create three problems for document question answering over sensitive or domain-specific material:

- **Privacy.** Uploading internal documents to a third-party service may be prohibited or undesirable.
- **Connectivity and cost.** API-based systems fail without reliable internet and incur per-request costs.
- **Hallucination.** General-purpose models answer from training data rather than from the user's documents.

EdgeRAG addresses these by keeping all data and computation on the device and constraining the model to answer from retrieved passages only.

## Key Features

- Fully offline inference: no external API calls once models are downloaded
- Multi-PDF ingestion with page-level metadata
- Source citations (filename and page) attached to every answer
- Retrieved-context viewer, so users can inspect exactly which passages the model saw
- Persistent local vector index across sessions
- Identical pipeline on development (macOS) and target (Jetson Nano) hardware, with a single swappable LLM module
- Built-in benchmarking script for latency and memory measurement

## System Architecture

```
PDF upload -> text extraction -> chunking -> embedding -> FAISS index
                                                              |
User question -> embed query -> vector search -> top-k chunks
                                                              |
                                          local LLM -> grounded answer + citations
```

| Stage | Description |
|---|---|
| Ingestion | PDFs are parsed page by page with PyMuPDF; each page keeps filename and page number as metadata |
| Chunking | Text is split into 500-character chunks using LangChain's recursive splitter |
| Embedding | Each chunk is encoded as a 384-dimensional vector with `all-MiniLM-L6-v2` |
| Indexing | Vectors are stored in a local FAISS index (CPU), persisted to disk |
| Retrieval | The query is embedded and the top 3 chunks are retrieved by vector similarity |
| Generation | The local LLM receives the retrieved chunks and the question, and returns an answer |

## Tech Stack

| Component | Development (macOS) | Deployment (Jetson Nano) |
|---|---|---|
| PDF parsing | PyMuPDF | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter | Same |
| Embeddings | all-MiniLM-L6-v2 (about 80 MB) | Same |
| Vector store | FAISS (CPU) | FAISS (CPU) |
| LLM runtime | Ollama | llama.cpp with CUDA |
| LLM | TinyLlama 1.1B | TinyLlama 1.1B Chat, Q4_K_M (GGUF) |
| Interface | Streamlit | Streamlit |

Ollama does not support the Jetson Nano's ARM64 and Maxwell GPU combination, so the Jetson build uses llama.cpp compiled directly with CUDA support.

## Repository Structure

```
offline_rag/
├── ingestion/
│   ├── parser.py          # PDF -> pages with metadata
│   └── chunker.py         # Pages -> chunks
├── embeddings/
│   └── embedder.py        # Text -> 384-dim vectors
├── vectorstore/
│   └── store.py           # FAISS index: add, search, save, load
├── llm/
│   └── local_llm.py       # LLM wrapper (Ollama; replaced by llama.cpp on Jetson)
├── rag/
│   └── pipeline.py        # Retrieval and generation chain
├── eval/
│   └── measure.py         # Latency, RAM, and GPU benchmarking
├── app.py                 # Streamlit interface
└── requirements.txt
```

## Installation and Usage

### Requirements

- Python 3.9 or later (macOS development environment)
- [Ollama](https://ollama.com) installed locally

### Setup

```bash
git clone https://github.com/shrushti425/offline_rag.git
cd offline_rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull tinyllama
```

### Run

```bash
# Terminal 1
ollama serve

# Terminal 2
streamlit run app.py
```

Open `http://localhost:8501`, then:

1. Upload one or more PDFs from the sidebar.
2. Select **Index Documents** and wait for indexing to complete.
3. Enter a question and select **Ask**.
4. Expand **See retrieved context** to inspect the passages used.

The index is saved to `vectorstore.pkl` and reloaded on the next session. **Clear Index** resets it.

## Deployment on NVIDIA Jetson Nano

**Target environment:** Jetson Nano 4 GB, JetPack 4.6.x (Ubuntu 18.04).

### 1. Build llama.cpp with CUDA

```bash
sudo apt update && sudo apt install cmake build-essential python3-pip git -y

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
make -j4

pip3 install llama-cpp-python
```

### 2. Download a quantised model

```bash
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -O models/tinyllama.gguf
```

### 3. Replace the LLM wrapper

Replace `llm/local_llm.py` with:

```python
from llama_cpp import Llama

class LocalLLM:
    def __init__(self, model_path="models/tinyllama.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=20,   # layers offloaded to the Jetson GPU
            n_threads=4,       # Jetson Nano has 4 CPU cores
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["</s>", "User:", "Question:"],
        )
        return output["choices"][0]["text"].strip()
```

No other module changes. If the device runs out of memory, reduce `n_gpu_layers` or use a smaller model.

## Evaluation

### Methodology

| Metric | Method |
|---|---|
| End-to-end latency | Wall-clock time per query, reported separately for retrieval and generation |
| Peak memory | Resident memory sampled with `psutil` during a query |
| Power draw | `tegrastats` on the Jetson Nano |
| Retrieval quality | Hit rate at k=3: fraction of questions where at least one retrieved chunk contains the answer |
| Answer quality | Manual grading against a fixed question set (correct, partially correct, incorrect) |

**Test set:** `<number>` questions over `<number>` documents (`<document types and total pages>`).

### Results

| Metric | macOS (Ollama) | Jetson Nano (llama.cpp, Q4_K_M) |
|---|---|---|
| Mean latency (s) | `<measured>` | `<measured>` |
| Retrieval latency (s) | `<measured>` | `<measured>` |
| Generation latency (s) | `<measured>` | `<measured>` |
| Peak RAM (GB) | `<measured>` | `<measured>` |
| Mean power (W) | n/a | `<measured>` |
| Retrieval hit rate @ 3 | `<measured>` | `<measured>` |
| Answer accuracy | `<measured>` | `<measured>` |

### Reproducing the benchmarks

```bash
python eval/measure.py
```

Run the benchmark on the Jetson Nano and on the development machine against the same question set.

## Limitations

- TinyLlama (1.1B parameters) has limited reasoning ability; answer quality degrades on questions that need synthesis across several passages.
- Only text-based PDFs are supported; scanned documents require OCR, which is not included.
- Retrieval uses fixed-size character chunks and dense retrieval only, with no reranking.
- The index is stored as a single pickle file and is not designed for large corpora.

## Future Work

- Quantitative comparison of chunk size, overlap, and top-k settings
- Hybrid retrieval (dense and keyword) with a lightweight reranker
- Evaluation of alternative small models (for example Phi-2 and Qwen-1.5B) under the same memory budget
- OCR support for scanned documents
- Role-based access control for multi-user deployments

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://langchain.com)
- [Streamlit](https://streamlit.io)
- [PyMuPDF](https://pymupdf.readthedocs.io)

## Author

Shrushti Raut
[GitHub](https://github.com/shrushti425)
*Capstone Project — Edge AI + RAG + Jetson Nano*

