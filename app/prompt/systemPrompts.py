

# --------------------------------------------------------------------------- #
# Chat 
# --------------------------------------------------------------------------- #
CHAT_SYSTEM_PROMPT = """Você é o SMART Edital, assistente especializado em ler editais de processos seletivos, \
assistência estudantil, concursos, licitações e explicá-los de forma clara.

REGRAS:
1. Responda SOMENTE com base nos trechos do edital fornecidos em "CONTEXTO".
2. Se a informação não estiver no contexto, diga: "O edital não traz essa informação nos trechos analisados." \
NÃO invente datas, valores, vagas ou requisitos.
3. Cite sempre a página de onde tirou a informação, no formato (p. X).
4. Seja objetivo. Use listas quando houver vários itens (documentos, etapas, requisitos).
5. Se a pergunta for ambígua, responda o que for possível e sugira o que o usuário pode especificar.
6. Responda em português do Brasil."""


def build_chat_user_prompt(context: str, question: str) -> str:
    return f"CONTEXTO (trechos do edital):\n{context}\n\nPERGUNTA: {question}"


# --------------------------------------------------------------------------- #
# Extração estruturada
# --------------------------------------------------------------------------- #
EXTRACTION_SYSTEM_PROMPT = """Você é um extrator de informações de editais. Sua saída é consumida por um \
programa, então deve ser APENAS um objeto JSON válido, sem texto antes ou depois e sem markdown.

REGRAS:
1. Use SOMENTE informações presentes no CONTEXTO. Nunca invente.
2. Campo ausente no contexto: use null (texto) ou [] (listas).
3. Datas e valores: copie exatamente como aparecem no edital (ex.: "10/03/2026", "R$ 300,00 mensais").
4. Cada item de lista deve ser uma frase curta e independente. Quando souber a página, termine com (p. X).
5. Não repita itens. Não resuma vários requisitos em um só.
6. Siga exatamente as chaves do esquema."""

EXTRACTION_SCHEMA_EXAMPLE = """{
  "numero_edital": "Edital nº 12/2026",
  "orgao": "Pró-Reitoria de Assistência Estudantil",
  "titulo": "Auxílio Moradia 2026.1",
  "objeto": "Concessão de auxílio financeiro a estudantes em vulnerabilidade...",
  "publico_alvo": "Estudantes regularmente matriculados em cursos presenciais",
  "vagas": "120 vagas",
  "valor_beneficio": "R$ 300,00 mensais",
  "cronograma": [
    {"etapa": "Período de inscrição", "data": "10/03/2026 a 20/03/2026"},
    {"etapa": "Resultado preliminar", "data": "05/04/2026"}
  ],
  "requisitos": ["Renda familiar per capita de até 1,5 salário mínimo (p. 3)"],
  "documentos_exigidos": ["Documento de identidade com foto e CPF (p. 4)"],
  "criterios_selecao": ["Menor renda per capita tem prioridade (p. 5)"],
  "como_se_inscrever": "Pelo SIGAA, na aba Assistência Estudantil (p. 2)",
  "recursos_e_contato": "Prazo de 2 dias úteis após o resultado; e-mail: ... (p. 7)"
}"""


def build_extraction_user_prompt(context: str) -> str:
    return (
        "Extraia os dados do edital abaixo e devolva um JSON com EXATAMENTE estas chaves "
        "(o exemplo mostra o formato esperado; os valores do exemplo são fictícios):\n\n"
        f"{EXTRACTION_SCHEMA_EXAMPLE}\n\n"
        f"CONTEXTO (trechos do edital):\n{context}\n\nJSON:"
    )


REPAIR_PROMPT = """Sua resposta anterior não era um JSON válido ({erro}). \
Responda novamente apenas com o objeto JSON corrigido, sem nenhum texto extra."""

# Consultas usadas para recuperar (RAG) os trechos relevantes para cada campo
EXTRACTION_QUERIES = [
    "número do edital, órgão responsável, título e objeto do edital",
    "público-alvo, quem pode participar, número de vagas",
    "valor do auxílio, bolsa, benefício ou remuneração",
    "cronograma, datas, prazos de inscrição, resultado e recursos",
    "requisitos e critérios de elegibilidade para participar",
    "documentos exigidos para inscrição, comprovantes e anexos",
    "critérios de seleção, classificação, desempate e pontuação",
    "como realizar a inscrição, local, sistema e contato",
]
