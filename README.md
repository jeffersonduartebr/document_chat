# Document Chat & PRISMA Review App

An interactive Streamlit application for uploading one or more PDF/TXT documents, querying their content via embeddings + retrieval, and—using the PRISMA methodology—automatically generating a structured systematic literature review. Built with:

- **Docling** for document chunking and embeddings  
- **Ollama** as the LLM backend (HTTP API)  
- **Streamlit** for UI  
- **Docker & NVIDIA CUDA** for containerized, GPU-accelerated deployment  

---

## 1. Key Benefits

- **Multi-document Q&A**  
  Upload multiple files at once, retrieve the top-k most relevant passages, and get instant answers to your questions.  
- **Configurable LLM**  
  Select your Ollama model, adjust temperature, and choose between multiple embedding backends—all via sidebar controls.  
- **Two-layer Caching**  
  - **In-memory cache** for repeat query acceleration  
  - **Persistent disk cache** for embeddings across restarts  
- **PRISMA-style Review Generator**  
  One click produces a PRISMA flowchart (identification, screening, eligibility, inclusion), Methods, Results, Discussion sections, plus a bibliography.  
- **Containerized GPU-ready**  
  Leverages NVIDIA CUDA in Docker Compose for hardware-accelerated inference; scales Ollama independently from the front-end.

---

## 2. Challenges & Considerations

- **Context-window limits**  
  Very large documents may exceed the model’s max context. Pre-filter or split your corpus if necessary.  
- **Cache invalidation**  
  When documents or models change, you must clear `.cache_indices/` to rebuild embeddings.  
- **Service dependency**  
  Requires a running Ollama HTTP endpoint—minor network latency applies.  
- **Resource usage**  
  Embedding and inference are memory- and GPU-intensive; monitor utilization in production.  
- **PRISMA accuracy**  
  Generated reviews provide a strong draft but should be proofread for methodological rigor.

---

## 3. Installation & Usage

### 3.1 Prerequisites

- **Docker ≥ 20.10** & **Docker Compose ≥ 1.28**  
- **NVIDIA Container Toolkit** for GPU support  
- (Optional) **Python 3.11+** & `pip` for local runs  

### 3.2 Clone the Repository

```bash
git clone https://github.com/your-org/doc-chat-prisma.git
cd doc-chat-prisma
