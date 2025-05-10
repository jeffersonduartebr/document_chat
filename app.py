import os
import io
import hashlib
import pickle
import streamlit as st
import requests
from typing import List

from docling import Document
from docling.embedding import EmbeddingModel
from docling.retrieval import SimpleRetriever

# pasta para cache dos índices
CACHE_DIR = ".cache_indices"
os.makedirs(CACHE_DIR, exist_ok=True)

# configurações do endpoint Ollama a partir do ambiente
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1/chat/completions"

def hash_file_content(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def load_or_build_retriever(
    file_bytes_list: List[bytes],
    file_names: List[str],
    embed_model: str,
    use_cache: bool
):
    combined = "".join([hash_file_content(b) for b in file_bytes_list] + [embed_model]).encode("utf-8")
    key = hashlib.md5(combined).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            retriever, docs = pickle.load(f)
        return retriever, docs

    docs = []
    for content, name in zip(file_bytes_list, file_names):
        tmp_path = os.path.join(CACHE_DIR, name)
        with open(tmp_path, "wb") as f:
            f.write(content)
        docs.extend(Document.load(tmp_path))

    embedder = EmbeddingModel(embed_model)
    embeddings = embedder.embed_documents(docs)
    retriever = SimpleRetriever(embeddings, docs)

    with open(cache_path, "wb") as f:
        pickle.dump((retriever, docs), f)

    return retriever, docs

def query_ollama(prompt: str, model: str, temperature: float) -> str:
    """
    Chama o endpoint HTTP do Ollama usando formato compatível com OpenAI Chat API.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def main():
    st.set_page_config(page_title="📚 Chat & Revisão PRISMA", layout="wide")
    st.title("🔍💬 Interaja com Documentos  + 📑 Revisão Sistemática (PRISMA)")

    # cache em memória de respostas Ollama
    if "ollama_cache" not in st.session_state:
        st.session_state.ollama_cache = {}

    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    ollama_model = st.sidebar.selectbox("Modelo Ollama", ["llama2", "vicuna", "custom-model"])
    temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.7, step=0.05)
    embed_model = st.sidebar.selectbox("Modelo de Embeddings", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
    top_k = st.sidebar.slider("Número de trechos (k)", 1, 10, 3)
    use_cache = st.sidebar.checkbox("Usar cache de embeddings", value=True)
    st.sidebar.markdown("---")
    st.sidebar.write("Reenvie o mesmo documento para reutilizar embeddings salvos.")

    # Upload múltiplo
    uploaded_files = st.file_uploader("📄 Carregue PDFs/TXTs", type=["pdf", "txt"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Faça o upload de pelo menos um documento para começar.")
        return

    file_bytes_list = [uf.read() for uf in uploaded_files]
    file_names = [uf.name for uf in uploaded_files]

    with st.spinner("Processando documento(s)…"):
        retriever, docs = load_or_build_retriever(file_bytes_list, file_names, embed_model, use_cache)
    st.success("Documento(s) pronto(s) para consulta!")

    # Chat simples
    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("💬 Chat sobre o documento")
    query = st.text_input("Pergunta:")
    if st.button("Enviar pergunta"):
        top_chunks = retriever.get_relevant(query, k=top_k)
        contexto = "\n\n---\n\n".join([c.content for c in top_chunks])
        prompt = f"Contexto:\n{contexto}\n\nPergunta: {query}\nResposta:"
        key = hashlib.md5((prompt + ollama_model + str(temperature)).encode()).hexdigest()
        if key in st.session_state.ollama_cache:
            resposta = st.session_state.ollama_cache[key]
        else:
            resposta = query_ollama(prompt, ollama_model, temperature)
            st.session_state.ollama_cache[key] = resposta

        st.markdown(f"**Resposta:** {resposta}")
        st.session_state.history.append((query, resposta))

    if st.checkbox("Mostrar histórico de chat"):
        for i, (q, a) in enumerate(st.session_state.history, 1):
            st.markdown(f"**{i}. Q:** {q}  \n**A:** {a}")

    # Revisão sistemática PRISMA
    st.markdown("---")
    st.subheader("📑 Gerar Revisão Sistemática (PRISMA)")
    if st.button("Gerar revisão PRISMA"):
        with st.spinner("Gerando revisão sistemática..."):
            all_text = "\n\n".join([c.content for c in docs])
            prompt = f"""
Você é um pesquisador especialista em Revisões Sistemáticas de Literatura, seguindo a metodologia PRISMA.
Com base exclusivamente nos documentos carregados, produza:

1. Fluxograma PRISMA com:
   - Identificados
   - Triados
   - Elegíveis
   - Incluídos

2. Seções Métodos, Resultados e Discussão.

3. Lista de referências.

Documentos:
{all_text}

Resposta em texto corrido e tabelas simples.
"""
            review = query_ollama(prompt, ollama_model, temperature)
            st.text_area("Revisão Sistemática (PRISMA)", review, height=400)

if __name__ == "__main__":
    main()
