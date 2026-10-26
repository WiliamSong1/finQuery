
import requests
import streamlit as st


st.set_page_config(
    page_title="Document Research | FinQuery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------
# Styling
# ------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0c111b;
        color: #edf1f7;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: 2.25rem !important;
        font-weight: 720 !important;
    }

    h2 {
        font-size: 1.3rem !important;
        font-weight: 650 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #4778eb;
        border: none;
        border-radius: 9px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #6490fa;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #293244;
        border-radius: 12px;
        padding: 0.9rem;
        margin-bottom: 0.6rem;
    }

    [data-testid="stExpander"] {
        border: 1px solid #293244;
        border-radius: 10px;
    }

    .fq-eyebrow {
        color: #8aa7f8;
        font-size: 0.77rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
    }

    .fq-subtitle {
        color: #a0aec3;
        font-size: 0.98rem;
        margin-bottom: 1.5rem;
    }

    .fq-stat {
        border: 1px solid #283348;
        border-radius: 10px;
        padding: 14px 16px;
        background: #141c2a;
    }

    .fq-stat-title {
        color: #9aa8bc;
        font-size: 0.78rem;
    }

    .fq-stat-value {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------
# API configuration
# ------------------------------------

API_URL = st.secrets["FINQUERY_API_URL"].rstrip("/")
API_KEY = st.secrets["FINQUERY_API_KEY"]

HEADERS = {"X-API-Key": API_KEY}

if "research_messages" not in st.session_state:
    st.session_state.research_messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = False


def check_backend():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=10,
        )
        response.raise_for_status()

        return response.json().get("status") == "healthy"
    except (requests.RequestException, ValueError):
        return False


def get_answer(question):
    response = requests.post(
        f"{API_URL}/agent/ask",
        headers=HEADERS,
        json={"query": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# ------------------------------------
# Sidebar
# ------------------------------------

with st.sidebar:
    st.title("FinQuery")
    st.caption("Financial intelligence workspace")
    st.divider()

    st.markdown("**Workspace**")
    st.caption("Use the page navigation above to switch between financial analysis and document research.")

    st.divider()

    st.markdown("**Session**")

    if st.button("New research session", use_container_width=True):
        st.session_state.research_messages = []
        st.session_state.uploaded_names = []
        st.rerun()

    st.caption(
        "New session clears this browser's research history. "
        "It does not delete PDFs from the AWS database."
    )

    st.divider()

    with st.expander("About FinQuery"):
        st.caption(
            "Financial document research powered by "
            "FastAPI, Qdrant, LangGraph, and Groq."
        )
        st.caption("Research use only. Verify material financial decisions.")


# ------------------------------------
# Header
# ------------------------------------

st.markdown(
    '<div class="fq-eyebrow">RESEARCH WORKSPACE</div>',
    unsafe_allow_html=True,
)

st.title("Document Research")

st.markdown(
    '<div class="fq-subtitle">'
    'Ask questions across SEC filings with '
    'retrieval-backed answers and source citations.'
    '</div>',
    unsafe_allow_html=True,
)

st.session_state.backend_ok = check_backend()

a, b, c = st.columns(3)

with a:
    st.markdown(
        '<div class="fq-stat">'
        '<div class="fq-stat-title">Backend</div>'
        '<div class="fq-stat-value">'
        + ("● Connected" if st.session_state.backend_ok
           else "● Offline")
        + '</div></div>',
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        '<div class="fq-stat">'
        '<div class="fq-stat-title">Uploaded this session</div>'
        f'<div class="fq-stat-value">'
        f'{len(st.session_state.uploaded_names)} documents'
        '</div></div>',
        unsafe_allow_html=True,
    )

with c:
    st.markdown(
        '<div class="fq-stat">'
        '<div class="fq-stat-title">Research engine</div>'
        '<div class="fq-stat-value">Hybrid RAG</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.write("")


# ------------------------------------
# Upload section
# ------------------------------------

with st.expander("Upload financial filings", expanded=True):
    st.caption(
        "Add 10-K or 10-Q PDFs to FinQuery's document index."
    )

    st.caption(
        "Private testing only: the backend currently shares "
        "indexed documents across sessions. "
        "Upload public, non-sensitive filings."
    )

    uploaded_files = st.file_uploader(
        "PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button(
        "Process documents",
        type="primary",
        disabled=not st.session_state.backend_ok,
    ):
        if not uploaded_files:
            st.warning("Select at least one PDF.")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(
                    f"Indexing {uploaded_file.name}..."
                ):
                    try:
                        response = requests.post(
                            f"{API_URL}/documents",
                            headers=HEADERS,
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf",
                                )
                            },
                            timeout=180,
                        )
                        response.raise_for_status()

                        if uploaded_file.name not in (
                            st.session_state.uploaded_names
                        ):
                            st.session_state.uploaded_names.append(
                                uploaded_file.name
                            )

                        st.success(
                            f"Indexed {uploaded_file.name}"
                        )

                        with st.expander(
                            f"Details: {uploaded_file.name}"
                        ):
                            st.json(response.json())

                    except requests.RequestException as exc:
                        st.error(
                            f"Upload failed for "
                            f"{uploaded_file.name}: {exc}"
                        )


# ------------------------------------
# Research chat
# ------------------------------------

st.write("")
st.subheader("Ask FinQuery")

st.caption(
    "Ask about revenue, operating expenses, "
    "management commentary, or year-over-year changes."
)

if not st.session_state.research_messages:
    st.info(
        "Start with a question such as: "
        "How did Apple's operating expenses change "
        "between 2023 and 2025?"
    )


def show_sources(sources):
    if not sources:
        return

    with st.expander(
        f"Supporting sources ({len(sources)})"
    ):
        for source in sources:
            citation_id = source.get("citation_id", "?")
            filename = source.get("document", "Unknown filing")
            page = source.get("page", "?")

            st.markdown(
                f"**[{citation_id}] {filename}** — Page {page}"
            )

            if source.get("text"):
                st.caption(source["text"][:1000])

            st.divider()


def show_response(result):
    answer = result.get("answer", "")
    status = result.get("status", "unknown")

    if answer:
        st.markdown(answer)
    else:
        st.warning("The backend did not return an answer.")

    if status != "answered":
        st.caption(f"Result status: {status}")

    show_sources(result.get("sources", []))

    with st.expander("Technical details"):
        st.json(result)


for message in st.session_state.research_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_response(message["result"])
        else:
            st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about financial filings...",
    disabled=not st.session_state.backend_ok,
)

if question:
    st.session_state.research_messages.append({
        "role": "user",
        "content": question,
    })

    with st.spinner("Searching filings and verifying evidence..."):
        try:
            result = get_answer(question)

        except requests.RequestException as exc:
            result = {
                "status": "request_failed",
                "answer": (
                    "FinQuery could not complete this request. "
                    "Check the backend and try again."
                ),
                "error": str(exc),
                "sources": [],
            }

    st.session_state.research_messages.append({
        "role": "assistant",
        "result": result,
    })

    st.rerun()
