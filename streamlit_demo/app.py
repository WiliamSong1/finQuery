
from datetime import date
from decimal import Decimal

import requests
import streamlit as st


st.set_page_config(
    page_title="FinQuery",
    page_icon="📊",
    layout="wide"
)

APPLE_CIK = "0000320193"

METRICS = {
    "Operating Expenses": "OperatingExpenses",
    "Operating Income": "OperatingIncomeLoss",
    "Net Income": "NetIncomeLoss"
}


@st.cache_data(ttl=3600)
def load_company_facts():
    url = (
        "https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{APPLE_CIK}.json"
    )

    contact = st.secrets["SEC_USER_AGENT"]

    response = requests.get(
        url,
        headers={
            "User-Agent": contact,
            "Accept": "application/json"
        },
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def get_annual_value(facts, tag, year):
    concepts = facts["facts"]["us-gaap"]

    if tag not in concepts:
        raise ValueError(
            f"Financial metric {tag} unavailable"
        )

    observations = concepts[tag]["units"]["USD"]

    candidates = []

    for item in observations:
        if item.get("form") != "10-K":
            continue

        if item.get("fp") != "FY":
            continue

        if "start" not in item or "end" not in item:
            continue

        start = date.fromisoformat(item["start"])
        end = date.fromisoformat(item["end"])

        duration = (end - start).days

        if end.year != year:
            continue

        if not 330 <= duration <= 400:
            continue

        candidates.append(item)

    if not candidates:
        raise ValueError(
            f"No annual financial data for {year}"
        )

    candidates.sort(
        key=lambda x: (
            x.get("fy") != year,
            x.get("filed", "9999-99-99")
        )
    )

    return candidates[0]


st.title("FinQuery")
st.caption(
    "Financial intelligence powered by SEC filings"
)

st.markdown(
    "Explore reported financial results and compare "
    "fiscal years using official SEC XBRL data."
)

st.divider()

left, middle, right = st.columns(3)

with left:
    company = st.selectbox(
        "Company",
        ["Apple Inc."]
    )

with middle:
    metric = st.selectbox(
        "Financial metric",
        list(METRICS.keys())
    )

with right:
    start_year, end_year = st.select_slider(
        "Fiscal years",
        options=[2022, 2023, 2024, 2025],
        value=(2023, 2025)
    )

analyze = st.button(
    "Analyze financial performance",
    type="primary",
    use_container_width=True
)

if analyze:
    if start_year >= end_year:
        st.error("Choose two different fiscal years.")

    else:
        try:
            with st.spinner("Retrieving SEC data..."):
                facts = load_company_facts()
                tag = METRICS[metric]

                start = get_annual_value(
                    facts, tag, start_year
                )

                end = get_annual_value(
                    facts, tag, end_year
                )

                start_value = Decimal(
                    str(start["val"])
                )

                end_value = Decimal(
                    str(end["val"])
                )

                change = end_value - start_value

                percentage = (
                    change / abs(start_value) * 100
                    if start_value != 0
                    else None
                )

            st.subheader("Financial comparison")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    str(start_year),
                    f"${start_value / Decimal('1e9'):,.2f}B"
                )

            with c2:
                st.metric(
                    str(end_year),
                    f"${end_value / Decimal('1e9'):,.2f}B"
                )

            with c3:
                st.metric(
                    "Percentage change",
                    (
                        f"{percentage:+.2f}%"
                        if percentage is not None
                        else "N/A"
                    )
                )

            st.subheader("Analysis")

            direction = (
                "increased" if change > 0
                else "decreased" if change < 0
                else "remained unchanged"
            )

            st.write(
                f"Apple's {metric.lower()} {direction} "
                f"from ${start_value / Decimal('1e9'):,.3f} "
                f"billion in fiscal {start_year} to "
                f"${end_value / Decimal('1e9'):,.3f} "
                f"billion in fiscal {end_year}."
            )

            st.write(
                f"Absolute change: "
                f"${abs(change) / Decimal('1e9'):,.3f} billion."
            )

            if percentage is not None:
                st.write(
                    f"Percentage change: {percentage:+.2f}%."
                )

            st.subheader("SEC sources")

            for year, observation in [
                (start_year, start),
                (end_year, end)
            ]:
                with st.expander(f"Fiscal {year} filing"):
                    st.write(
                        "Accession:",
                        observation["accn"]
                    )
                    st.write(
                        "Period:",
                        observation["start"],
                        "to",
                        observation["end"]
                    )
                    st.write(
                        "Filed:",
                        observation["filed"]
                    )

            st.link_button(
                "View official SEC data",
                (
                    "https://data.sec.gov/api/xbrl/"
                    f"companyfacts/CIK{APPLE_CIK}.json"
                )
            )

        except (
            ValueError,
            KeyError,
            requests.RequestException
        ) as exc:
            st.error(
                "Unable to retrieve the requested "
                "financial data. Please try again."
            )
            st.caption(str(exc))

st.divider()

st.caption(
    "FinQuery Lite · SEC financial data · "
    "For research and educational use"
)
