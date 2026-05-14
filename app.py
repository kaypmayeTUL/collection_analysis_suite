"""
Library Collection Dashboard v4.0
================================
Decision-Support Toolkit for Howard-Tilton Memorial Library

Tools:
  1. Collection Profiler — LC/subject distribution & coverage vs. use.
  2. Zero-Use Identifier — Identify low-ROI holdings. (placeholder)
  3. Acquisition Scorer — Evidence-based purchasing. (placeholder)

Updated: 2025 | Contact: Kay P Maye (kmaye@tulane.edu)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# =====================================================================
# CONFIG & THEMING
# =====================================================================

st.set_page_config(page_title="Library Collection Dashboard", page_icon="📚", layout="wide")

st.markdown("""
<style>
    :root { --tulane-green: #285C4D; --tulane-blue: #71C5E8; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: bold; font-size: 1.05rem; }
    .tool-header { color: #285C4D; border-bottom: 2px solid #285C4D; padding-bottom: 5px; margin-bottom: 15px; }
    .data-source-box { background-color: #f0f2f6; border-radius: 8px; padding: 15px; border-left: 5px solid #71C5E8; margin-bottom: 25px; }
    .usage-guide { background-color: #eef6f3; border-radius: 8px; padding: 15px; border-left: 5px solid #285C4D; margin-bottom: 10px; }
    .decision-box { background-color: #fff8e6; border-radius: 8px; padding: 15px; border-left: 5px solid #e6a800; margin-bottom: 20px; }
    .stButton>button { background-color: #285C4D; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LC PARSING & NORMALIZATION HELPERS
# =====================================================================

LC_MAIN_MAP = {
    "A": "General Works",
    "B": "Philosophy, Psychology, Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History",
    "E": "History of the Americas",
    "F": "History of the Americas (Local)",
    "G": "Geography, Anthropology, Recreation",
    "H": "Social Sciences",
    "J": "Political Science",
    "K": "Law",
    "L": "Education",
    "M": "Music",
    "N": "Fine Arts",
    "P": "Language & Literature",
    "Q": "Science",
    "R": "Medicine",
    "S": "Agriculture",
    "T": "Technology",
    "U": "Military Science",
    "V": "Naval Science",
    "Z": "Bibliography, Library Science, Information Resources"
}

def parse_lc_from_callnumber(callnum: str):
    """
    Parse LC main class and subclass from a call number string.
    Returns (main_class_code, subclass_code, main_desc, subclass_desc).
    """
    if not isinstance(callnum, str) or not callnum.strip():
        return ("Unclassified", "Unclassified", "Unclassified", "Unclassified")

    # Basic LC pattern: 1–3 letters at start
    m = re.match(r"^([A-Z]{1,3})", callnum.strip().upper())
    if not m:
        return ("Unclassified", "Unclassified", "Unclassified", "Unclassified")

    subclass_code = m.group(1)
    main_code = subclass_code[0]

    main_desc = LC_MAIN_MAP.get(main_code, "Other / Non-LC")
    # For now, subclass description is just the code; you can map further if desired
    subclass_desc = f"{subclass_code} – LC Subclass"

    return (main_code, subclass_code, main_desc, subclass_desc)


def normalize_bibliographic_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize different file types into a common structure for the Collection Profiler.

    Handles:
    - Alma physical inventory / bib exports (with call numbers)
    - Alma physical/digital usage exports
    - ProQuest / LibCentral title lists (like the sample you provided)
    """

    df_norm = pd.DataFrame(index=df.index)

    # ---------------- SUBJECT / INTELLECTUAL AREA ----------------
    # Try to capture some subject text for later use
    subject_col_candidates = [
        "Subject", "Subjects", "LC Subject", "LC_Subject", "Subject_Heading"
    ]
    subj_col = next((c for c in subject_col_candidates if c in df.columns), None)
    if subj_col:
        df_norm["Subject_Heading"] = df[subj_col].astype(str)
    else:
        df_norm["Subject_Heading"] = "Unspecified"

    # ---------------- CALL NUMBER → LC PARSING ----------------
    callnum_candidates = [
        "CallNumber", "Call Number", "Permanent Call Number",
        "LC_CallNumber", "LC Call Number", "Call_Number"
    ]
    call_col = next((c for c in callnum_candidates if c in df.columns), None)

    if call_col:
        lc_parsed = df[call_col].apply(parse_lc_from_callnumber)
        df_norm["LC_Main_Code"] = lc_parsed.apply(lambda x: x[0])
        df_norm["LC_Subclass_Code"] = lc_parsed.apply(lambda x: x[1])
        df_norm["Main_LC_Desc"] = lc_parsed.apply(lambda x: x[2])
        df_norm["Subclass_Desc"] = lc_parsed.apply(lambda x: x[3])
    else:
        # No call numbers: derive pseudo-LC from subject strings (e.g., ProQuest LibCentral)
        def derive_main(subject):
            if pd.isna(subject) or not str(subject).strip():
                return "Unclassified"
            # Take first segment before ';' or ':'
            return str(subject).split(";")[0].split(":")[0].strip()

        def derive_sub(subject):
            if pd.isna(subject) or not str(subject).strip():
                return "Unclassified"
            parts = str(subject).split(";")
            return parts[1].strip() if len(parts) > 1 else "General"

        df_norm["Main_LC_Desc"] = df_norm["Subject_Heading"].apply(derive_main)
        df_norm["Subclass_Desc"] = df_norm["Subject_Heading"].apply(derive_sub)
        df_norm["LC_Main_Code"] = df_norm["Main_LC_Desc"]
        df_norm["LC_Subclass_Code"] = df_norm["Subclass_Desc"]

    # ---------------- USAGE / DEMAND ----------------
    usage_candidates = [
        "Total Book Downloads",
        "Read Online (post Trigger) Sessions",
        "Total_Uses",
        "Total Uses",
        "Loans",
        "Checkouts",
        "Loan Count",
        "Usage"
    ]
    usage_col = next((c for c in usage_candidates if c in df.columns), None)
    if usage_col:
        df_norm["Usage_Count"] = pd.to_numeric(df[usage_col], errors="coerce").fillna(0)
    else:
        df_norm["Usage_Count"] = 0

    # ---------------- TITLE (OPTIONAL, FOR CONTEXT) ----------------
    title_candidates = ["Title", "Bib Title", "MMS Title", "title"]
    title_col = next((c for c in title_candidates if c in df.columns), None)
    if title_col:
        df_norm["Title"] = df[title_col].astype(str)
    else:
        df_norm["Title"] = ""

    return df_norm


def normalize_usage_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a usage-only file so it can be compared to the bibliographic file.
    We assume it has either call numbers or subjects.
    """
    df_norm = normalize_bibliographic_file(df)
    return df_norm


# =====================================================================
# HOME PAGE
# =====================================================================

def page_home():
    st.title("🏛️ Library Collection Decision Support Dashboard")

    st.markdown("""
    This dashboard consolidates three tools to support **collection development, weeding, and strategic purchasing**.
    Use the sidebar to navigate between tools.
    """)

    st.divider()

    tools = [
        ("🗺️", "Collection Profiler",
         "Analyze LC/subject distribution and compare coverage vs. use.",
         [
             "Where are we over- or under-collecting?",
             "Which LC/subject areas show unmet demand?",
             "Which subjects need liaison or programmatic attention?"
         ]),
        ("🔍", "Zero-Use Identifier",
         "Identify holdings with no recorded use to support deselection or storage decisions.",
         [
             "Which items are candidates for off-site storage?",
             "Which areas show persistent non-use?",
             "What is the ROI of specific donor or package collections?"
         ]),
        ("📊", "Acquisition Scorer",
         "Score candidate titles using local usage patterns and subject trends.",
         [
             "Which titles should we prioritize?",
             "How do vendor lists align with local needs?",
             "Where should end-of-year funds be focused?"
         ])
    ]

    for icon, title, desc, decisions in tools:
        st.markdown(f"<h3 class='tool-header'>{icon} {title}</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown(
                f"<div class='usage-guide'><strong>What it does:</strong><br>{desc}</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                "<div class='decision-box'><strong>Supports decisions such as:</strong><ul>" +
                "".join([f"<li>{d}</li>" for d in decisions]) +
                "</ul></div>",
                unsafe_allow_html=True
            )
        st.divider()


# =====================================================================
# TOOL 1: COLLECTION PROFILER
# =====================================================================

def page_collection_profiler():
    st.title("🗺️ Collection Profiler & Subject Analysis")

    st.markdown("""
    Use this tool to understand the **shape of your collection** and how it aligns with **actual use**.

    It can work with:
    - Alma physical inventory / bib exports (with call numbers)
    - Alma physical or digital usage exports
    - ProQuest / LibCentral title lists (like the sample you provided)
    """)

    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Display Settings")
        chart_height = st.slider("Chart Height", 400, 1400, 800)
        show_table = st.checkbox("Show Data Tables", True)

        st.header("📁 Data Files")
        st.caption("Upload at least one bibliographic or title list file.")
        bib_file = st.file_uploader("Title / Inventory / Vendor List CSV", type="csv")
        use_file = st.file_uploader("Usage / Circulation CSV (Optional)", type="csv")

    if not bib_file:
        st.info("Upload a bibliographic or title list file to begin (e.g., Alma inventory, ProQuest LibCentral).")
        return

    # Load and normalize bibliographic/title data
    df_raw_bib = pd.read_csv(bib_file)
    df_bib = normalize_bibliographic_file(df_raw_bib)
    st.success(f"Loaded {len(df_bib):,} records (auto-normalized).")

    # Optional usage file
    df_use = None
    if use_file:
        df_raw_use = pd.read_csv(use_file)
        df_use = normalize_usage_file(df_raw_use)
        st.info(f"Loaded {len(df_use):,} usage records (auto-normalized).")

    tab1, tab2, tab3 = st.tabs([
        "🏛️ LC / Subject Distribution",
        "🏷️ Subject Concentration",
        "📈 Coverage vs Use"
    ])

    # ---------------- TAB 1: LC / SUBJECT DISTRIBUTION ----------------
    with tab1:
        st.subheader("LC / Subject Distribution")

        st.markdown("""
        This view shows how your collection is distributed across **LC classes or subject groupings**.

        Use it to:
        - Spot **overbuilt** or **underbuilt** areas
        - Check alignment with **programs and curricula**
        - Identify areas for **liaison outreach** or **accreditation support**
        """)

        required_cols = ["Main_LC_Desc", "Subclass_Desc"]

        if not all(col in df_bib.columns for col in required_cols):
            st.error("Unable to generate LC/Subject distribution — required fields are missing after normalization.")
        else:
            df_bib_clean = df_bib[required_cols].fillna("Unclassified")

            fig = px.sunburst(
                df_bib_clean,
                path=["Main_LC_Desc", "Subclass_Desc"],
                height=chart_height,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_table:
                with st.expander("📋 LC / Subject Summary Table"):
                    summary = (
                        df_bib_clean
                        .groupby("Main_LC_Desc")
                        .size()
                        .reset_index(name="Title_Count")
                        .sort_values("Title_Count", ascending=False)
                    )
                    st.dataframe(summary, use_container_width=True)

    # ---------------- TAB 2: SUBJECT CONCENTRATION ----------------
    with tab2:
        st.subheader("Subject Heading Concentration")

        st.markdown("""
        This view highlights the **intellectual shape** of the collection:

        - Which subjects dominate?
        - Are there emerging or neglected areas?
        - Do subject patterns align with **academic programs** and **research strengths**?
        """)

        if "Subject_Heading" not in df_bib.columns:
            st.warning("No subject heading field was detected after normalization.")
        else:
            df_sub = df_bib.copy()
            df_sub["Subject_Heading"] = df_sub["Subject_Heading"].fillna("Unspecified")

            fig_sub = px.treemap(
                df_sub,
                path=["Subject_Heading"],
                height=chart_height,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_sub, use_container_width=True)

            if show_table:
                with st.expander("📋 Top Subjects by Title Count"):
                    subj_summary = (
                        df_sub["Subject_Heading"]
                        .value_counts()
                        .reset_index()
                        .rename(columns={"index": "Subject_Heading", "Subject_Heading": "Title_Count"})
                    )
                    st.dataframe(subj_summary.head(50), use_container_width=True)

    # ---------------- TAB 3: COVERAGE VS USE ----------------
    with tab3:
        st.subheader("Coverage vs Use (Supply vs Demand)")

        st.markdown("""
        Compare what the library **owns** (or has access to) vs. what users **actually use**.

        This helps identify:
        - **Overbuilt** areas with low demand
        - **High-demand** areas needing investment
        - Subjects where **digital** may outperform **print**
        """)

        if df_use is None:
            st.warning("Upload a usage file to enable Coverage vs Use comparisons.")
        else:
            # Aggregate holdings by Main_LC_Desc
            holdings = (
                df_bib
                .groupby("Main_LC_Desc")
                .size()
                .reset_index(name="Holdings_Count")
            )

            # Aggregate usage by Main_LC_Desc
            usage = (
                df_use
                .groupby("Main_LC_Desc")["Usage_Count"]
                .sum()
                .reset_index(name="Usage_Count")
            )

            merged = pd.merge(holdings, usage, on="Main_LC_Desc", how="outer").fillna(0)

            # Convert to percentages for comparison
            if merged["Holdings_Count"].sum() > 0:
                merged["Holdings_%"] = merged["Holdings_Count"] / merged["Holdings_Count"].sum() * 100
            else:
                merged["Holdings_%"] = 0

            if merged["Usage_Count"].sum() > 0:
                merged["Usage_%"] = merged["Usage_Count"] / merged["Usage_Count"].sum() * 100
            else:
                merged["Usage_%"] = 0

            merged = merged.sort_values("Usage_%", ascending=False)

            fig_gap = go.Figure(data=[
                go.Bar(
                    name="Collection %",
                    x=merged["Main_LC_Desc"],
                    y=merged["Holdings_%"],
                    marker_color="#71C5E8"
                ),
                go.Bar(
                    name="Usage %",
                    x=merged["Main_LC_Desc"],
                    y=merged["Usage_%"],
                    marker_color="#285C4D"
                )
            ])
            fig_gap.update_layout(
                height=chart_height,
                barmode="group",
                xaxis_title="LC / Subject Area",
                yaxis_title="Percent of Total"
            )
            st.plotly_chart(fig_gap, use_container_width=True)

            if show_table:
                with st.expander("📋 Coverage vs Use Table"):
                    st.dataframe(
                        merged[["Main_LC_Desc", "Holdings_Count", "Usage_Count", "Holdings_%", "Usage_%"]],
                        use_container_width=True
                    )


# =====================================================================
# PLACEHOLDERS FOR OTHER TOOLS
# =====================================================================

def page_zero_use():
    st.title("🔍 Zero-Use Identifier")
    st.info("Zero-Use Identifier functionality will be implemented here.")


def page_acquisition_scorer():
    st.title("📊 Acquisition Scorer")
    st.info("Acquisition Scorer functionality will be implemented here.")


# =====================================================================
# MAIN NAVIGATION
# =====================================================================

def main():
    with st.sidebar:
        st.title("📚 Dashboard Navigation")
        page = st.radio(
            "Choose a Tool:",
            [
                "🏠 Home",
                "🗺️ Collection Profiler",
                "🔍 Zero-Use Identifier",
                "📊 Acquisition Scorer"
            ]
        )
        st.divider()
        st.caption("Tulane University | Howard-Tilton Memorial Library")

    if page == "🏠 Home":
        page_home()
    elif page == "🗺️ Collection Profiler":
        page_collection_profiler()
    elif page == "🔍 Zero-Use Identifier":
        page_zero_use()
    elif page == "📊 Acquisition Scorer":
        page_acquisition_scorer()


if __name__ == "__main__":
    main()
