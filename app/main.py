import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from app.core.config import settings
from app.core.context import format_context
from app.core.extractor import (
    EditalExtractor,
    ExtractionError,
    load_saved,
    save_extraction,
)
from app.core.llmClient import LLMClient, LLMError
from app.core.pdfProcessor import process_pdf, save_upload
from app.core.ragEngine import RAGEngine
from app.prompt.systemPrompts import (
    CHAT_SYSTEM_PROMPT,
    build_chat_user_prompt,
)

st.set_page_config(page_title=settings.PROJECT_NAME, layout="wide")


# --------------------------------------------------------------------------- #
# Recursos pesados em cache (carregados uma vez por processo)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def get_rag() -> RAGEngine:
    return RAGEngine()


@st.cache_resource(show_spinner=False)
def get_llm() -> LLMClient:
    return LLMClient()


rag = get_rag()
try:
    llm = get_llm()
except LLMError as e:
    st.error(str(e))
    st.stop()


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
def reset_state():
    st.session_state.update(doc=None, edital=None, extraction_info=None, messages=[])


if "doc" not in st.session_state:
    reset_state()

# --------------------------------------------------------------------------- #
# Sidebar: upload + parâmetros fixos
# --------------------------------------------------------------------------- #
top_k = 10
temperature = 0.10

with st.sidebar:
    st.title(f" {settings.PROJECT_NAME}")
    st.caption("Extrator e assistente de editais com RAG + LLM")
    uploaded = st.file_uploader("Selecione o edital (PDF)", type=["pdf"])
    st.divider()
    st.caption(f"**LLM:** {llm.describe()}")
    st.caption(f"**Embeddings:** {settings.EMBEDDING_MODEL}")
    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if uploaded is not None:
    key = (uploaded.name, uploaded.size)
    if st.session_state.get("file_key") != key:
        reset_state()
        path, doc_id = save_upload(uploaded.getvalue(), uploaded.name)
        with st.spinner("Lendo o PDF e dividindo em trechos..."):
            proc = process_pdf(path, doc_id, uploaded.name)
        if proc.is_scanned:
            st.error("Este PDF parece ser escaneado (sem texto). Use um PDF com texto selecionável ou aplique OCR antes.")
            st.stop()
        with st.spinner("Gerando embeddings e salvando no ChromaDB..."):
            created = rag.index_document(doc_id, proc.chunks)
        st.session_state.file_key = key
        st.session_state.doc = proc
        st.session_state.edital = load_saved(doc_id)  # reaproveita extração anterior
        st.session_state.indexed_now = created

doc = st.session_state.doc
if doc is None:
    st.title(settings.PROJECT_NAME)
    st.info("Envie um edital em PDF na barra lateral para começar.")
    st.stop()

st.subheader(doc.filename)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Páginas", doc.n_pages)
c2.metric("Caracteres", f"{doc.n_chars:,}".replace(",", "."))
c3.metric("Chunks", len(doc.chunks))
c4.metric("Índice vetorial", "novo" if st.session_state.get("indexed_now") else "reutilizado")

tab_ext, tab_chat = st.tabs(
    ["Extração estruturada", " Perguntar ao edital"]
)

# --------------------------------------------------------------------------- #
# Aba 1: extração
# --------------------------------------------------------------------------- #
with tab_ext:
    if st.button("Extrair dados do edital", type="primary"):
        try:
            with st.spinner("Recuperando trechos relevantes e consultando o LLM..."):
                result = EditalExtractor(rag, llm).extract(doc.doc_id)
            save_extraction(doc.doc_id, result.edital)
            st.session_state.edital = result.edital
            st.session_state.extraction_info = result
        except (ExtractionError, LLMError) as e:
            st.error(str(e))

    ed = st.session_state.edital
    if ed is None:
        st.info("Clique em **Extrair dados do edital** para gerar o resumo estruturado.")
    else:
        info = st.session_state.extraction_info
        if info:
            st.caption(f"Baseado em {info.n_chunks} trechos (páginas {', '.join(map(str, info.pages))}).")
        st.markdown(f"### {ed.titulo or 'Edital'}  \n**{ed.numero_edital or ''}** — {ed.orgao or ''}")
        if ed.objeto:
            st.write(ed.objeto)

        a, b, c = st.columns(3)
        a.markdown(f"**Público-alvo**  \n{ed.publico_alvo or '—'}")
        b.markdown(f"**Vagas**  \n{ed.vagas or '—'}")
        c.markdown(f"**Valor / benefício**  \n{ed.valor_beneficio or '—'}")

        st.markdown("#### Cronograma")
        if ed.cronograma:
            df = pd.DataFrame([e.model_dump() for e in ed.cronograma]).rename(columns={"etapa": "Etapa", "data": "Data"})
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("Nenhuma data encontrada.")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Requisitos")
            for r in ed.requisitos or ["—"]:
                st.markdown(f"- {r}")
            st.markdown("#### Critérios de seleção")
            for r in ed.criterios_selecao or ["—"]:
                st.markdown(f"- {r}")
        with col_r:
            st.markdown("#### Documentos exigidos")
            for r in ed.documentos_exigidos or ["—"]:
                st.markdown(f"- {r}")
            st.markdown("#### Inscrição")
            st.write(ed.como_se_inscrever or "—")
            st.markdown("#### Recursos e contato")
            st.write(ed.recursos_e_contato or "—")

        if ed.cronograma:
            st.download_button("⬇ Cronograma (CSV)", df.to_csv(index=False), f"{doc.doc_id}_cronograma.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Aba 2: chat RAG
# --------------------------------------------------------------------------- #
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Trechos usados"):
                    for s in m["sources"]:
                        st.markdown(f"**Página {s['page']}** · similaridade {s['score']:.2f}")
                        st.caption(s["text"])

    question = st.chat_input("Ex.: Quais documentos preciso apresentar? Qual o prazo de inscrição?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            results = rag.query(doc.doc_id, question, k=top_k)
            context = format_context(results, settings.MAX_CONTEXT_CHARS)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-2 * settings.HISTORY_TURNS :]
            ]
            messages = (
                [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
                + history
                + [{"role": "user", "content": build_chat_user_prompt(context, question)}]
            )
            try:
                answer = st.write_stream(llm.stream(messages, temperature=temperature))
            except LLMError as e:
                answer = f" {e}"
                st.error(answer)
            sources = [
                {"page": d.metadata["page"], "score": s, "text": d.page_content} for d, s in results
            ]
            with st.expander("Trechos usados"):
                for s in sources:
                    st.markdown(f"**Página {s['page']}** · similaridade {s['score']:.2f}")
                    st.caption(s["text"])
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})