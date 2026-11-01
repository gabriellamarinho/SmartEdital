import streamlit as st
from core.config import settings
from core.pdf_processor import process_pdf

st.set_page_config(page_title=settings.PROJECT_NAME, page_icon="📄", layout="wide")
st.title(f"{settings.PROJECT_NAME}")

if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None

uploaded_file = st.file_uploader("Selecione o edital (PDF)", type=["pdf"])

if uploaded_file and st.session_state.processed_pdf != uploaded_file.name:
    chunks = process_pdf(uploaded_file, source=uploaded_file.name)
    st.session_state.processed_pdf = uploaded_file.name
    st.success(f"{uploaded_file.name}: {len(chunks)} chunks gerados")
    st.write(chunks[0] if chunks else "Nenhum texto encontrado (PDF escaneado?)")