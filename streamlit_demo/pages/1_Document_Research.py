
import requests
import streamlit as st

st.set_page_config(
    page_title="FinQuery | Document Research",
    page_icon="📄",
    layout="wide",
)

API_URL = st.secrets["FINQUERY_API_URL"].rstrip("/")
API_KEY = st.secrets["FINQUERY_API_KEY"]

HEADERS = {
    "X-API-Key": API_KEY
}

st.title("Financial Document Research")
st.caption("Multi-document analysis powered by FinQuery on AWS")
st.warning(
    "Private testing only. Uploaded documents currently share "
    "the backend's Qdrant collection."
)

# -----------------------------
# Backend health check
# -----------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )
    health_response.raise_for_status()
    st.success("Connected to FinQuery on AWS")

except requests.RequestException:
    st.error("Cannot connect to the FinQuery backend.")
    st.stop()


# -----------------------------
# PDF uploads
# -----------------------------

st.header("1. Upload financial documents")

uploaded_files = st.file_uploader(
    "Choose 10-K or 10-Q PDF filings",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process documents", type="primary"):
    if not uploaded_files:
        st.warning("Select at least one PDF.")

    else:
        for uploaded_file in uploaded_files:
            if uploaded_file.size > 25 * 1024 * 1024:
                st.error(f"{uploaded_file.name} exceeds 25 MB.")
                continue

            with st.spinner(
                f"Processing {uploaded_file.name}..."
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

                    st.success(
                        f"Processed {uploaded_file.name}"
                    )

                    with st.expander("Ingestion details"):
                        st.json(response.json())

                except requests.RequestException as exc:
                    st.error(
                        f"Failed to process "
                        f"{uploaded_file.name}: {exc}"
                    )


# -----------------------------
# Financial questions
# -----------------------------

st.divider()
st.header("2. Ask FinQuery")

question = st.text_area(
    "Ask a question about the uploaded filings",
    placeholder=(
        "How did Apple's operating expenses change "
        "between 2023 and 2025, and why?"
    ),
    height=110,
)

if st.button("Ask FinQuery", type="primary"):
    if not question.strip():
        st.warning("Enter a question.")

    else:
        with st.spinner("FinQuery is analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/agent/ask",
                    headers=HEADERS,
                    json={"query": question.strip()},
                    timeout=180,
                )

                response.raise_for_status()
                result = response.json()

                st.subheader("Analysis")

                answer = result.get("answer")

                if isinstance(answer, str) and answer:
                    st.markdown(answer)
                else:
                    st.info(
                        "The agent returned a structured response."
                    )

                st.subheader("Full response")

                with st.expander(
                    "View citations and agent details",
                    expanded=True,
                ):
                    st.json(result)

            except requests.RequestException as exc:
                st.error(f"FinQuery request failed: {exc}")
