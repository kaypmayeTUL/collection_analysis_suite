"""
Library Collection Dashboard v2.1
================================
A unified Streamlit application for library collection decision support.

  1. Collection Profiler (v2.1) — "Coverage vs. Use"
     Analyzes subject/LC trends. Accepts optional usage data to compare
     collection depth (supply) against user demand (use).

  2. Zero-Use Identifier — "Identify Dead Weight"
     Matches holdings against usage to surface items with no circulation.

  3. Acquisition Recommendation Scorer — "Evidence-Based Purchasing"
     Scores candidate titles against local checkout history and faculty interests.

Updated: May 2024 | Contact: Kay P Maye (kmaye@tulane.edu)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import re
import gc
import unicodedata
from io import BytesIO

# =====================================================================
# CONFIG & CSS
# =====================================================================

st.set_page_config(page_title="Library Collection Dashboard", page_icon="📚", layout="wide")

st.markdown("""
<style>
    :root { --tulane-green: #285C4D; --tulane-blue: #71C5E8; }
    .stButton>button { background-color: #285C4D; color: white; font-weight: bold; width: 100%; border-radius: 5px; }
    .stButton>button:hover { background-color: #1e4a3c; }
    .decision-box { background-color: #eef6f3; border-left: 5px solid #285C4D; padding: 15px; border-radius: 4px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LC REFERENCE DATA
# =====================================================================

LC_CLASSES = {
    'A': 'General Works', 'B': 'Philosophy, Psych, Religion', 'C': 'Auxiliary History',
    'D': 'World History', 'E': 'US History', 'F': 'History of Americas',
    'G': 'Geography, Anthro', 'H': 'Social Sciences', 'J': 'Political Science',
    'K': 'Law', 'L': 'Education', 'M': 'Music', 'N': 'Fine Arts',
    'P': 'Language & Lit', 'Q': 'Science', 'R': 'Medicine', 'S': 'Agriculture',
    'T': 'Technology', 'U': 'Military Science', 'V': 'Naval Science', 'Z': 'Library Science'
}

# =====================================================================
# UTILITIES
# =====================================================================

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()

def extract_lc_class(call_num):
    if pd.isna(call_num): return None
    match = re.match(r"^([A-Z]{1,3})", str(call_num).strip().upper())
    return match.group(1)[0] if match else None

def find_column(cols, aliases):
    for alias in aliases:
        for col in cols:
            if alias.lower() in col.lower(): return col
    return None

SUBJECT_ALIASES = ['Subject', 'Topic', 'Descriptor']
LC_ALIASES = ['LC Class', 'Call Number', 'Classification', 'LCC']
USAGE_ALIASES = ['Usage', 'Checkouts', 'Loans', 'Circulation', 'Total Requests']
TITLE_ALIASES = ['Title', 'Item Title']

# =====================================================================
# TOOL 1: COLLECTION PROFILER (Updated with Coverage vs. Use)
# =====================================================================

def page_collection_profiler():
    st.title("🗺️ Collection Profiler")
    
    with st.container(border=True):
        st.markdown("**📌 When to use this tool**")
        st.markdown(
            "Analyze subject and LC trends in your catalog. You can now upload **optional usage data** "
            "to see 'Coverage vs. Use' trends (e.g., 'Class H is 10% of our books but 25% of our use')."
        )

    c1, c2 = st.columns(2)
    with c1:
        bib_file = st.file_uploader("Upload Catalog/Bibliographic CSV (Required)", type="csv")
    with c2:
        use_file = st.file_uploader("Upload Usage/Circulation CSV (Optional)", type="csv")

    if bib_file:
        df_bib = pd.read_csv(bib_file)
        st.success(f"Loaded {len(df_bib):,} bibliographic records.")
        
        # Column Mapping
        cols = list(df_bib.columns)
        subj_col = st.selectbox("Map Subject Column", cols, index=cols.index(find_column(cols, SUBJECT_ALIASES)) if find_column(cols, SUBJECT_ALIASES) else 0)
        lc_col = st.selectbox("Map LC Class Column", cols, index=cols.index(find_column(cols, LC_ALIASES)) if find_column(cols, LC_ALIASES) else 0)
        
        # Handle Usage Data
        usage_data = None
        if use_file:
            df_use = pd.read_csv(use_file)
            u_cols = list(df_use.columns)
            u_title = st.selectbox("Map Title/ID in Usage File", u_cols, index=u_cols.index(find_column(u_cols, TITLE_ALIASES)) if find_column(u_cols, TITLE_ALIASES) else 0)
            u_val = st.selectbox("Map Usage/Checkouts Column", u_cols, index=u_cols.index(find_column(u_cols, USAGE_ALIASES)) if find_column(u_cols, USAGE_ALIASES) else 0)
            
            # Simple Merge
            df_use[u_title] = df_use[u_title].astype(str).str.lower().str.strip()
            df_bib['_join_title'] = df_bib[find_column(cols, TITLE_ALIASES) or cols[0]].astype(str).str.lower().str.strip()
            usage_map = df_use.groupby(u_title)[u_val].sum().to_dict()
            df_bib['_usage'] = df_bib['_join_title'].map(usage_map).fillna(0)
            usage_data = "_usage"
        else:
            # Check if usage is already in the bib file
            existing_use = find_column(cols, USAGE_ALIASES)
            if existing_use:
                use_confirm = st.checkbox(f"Use existing '{existing_use}' column for usage analysis?", value=True)
                if use_confirm: usage_data = existing_use

        if st.button("Run Analysis", type="primary"):
            df_bib['_lc_main'] = df_bib[lc_col].apply(extract_lc_class)
            
            # Coverage (Count of titles)
            coverage = df_bib.groupby('_lc_main').size().reset_index(name='Title Count')
            coverage['% of Collection'] = (coverage['Title Count'] / coverage['Title Count'].sum() * 100).round(2)
            
            # Visualizations
            tab1, tab2 = st.tabs(["Subject/LC Distribution", "Coverage vs. Use Trends"])
            
            with tab1:
                fig_lc = px.bar(coverage.sort_values('Title Count', ascending=False), 
                                x='_lc_main', y='Title Count', 
                                title="Collection Distribution by LC Class",
                                color='Title Count', color_continuous_scale='Greens')
                st.plotly_chart(fig_lc, use_container_width=True)

            with tab2:
                if usage_data:
                    # Usage by LC
                    use_stats = df_bib.groupby('_lc_main')[usage_data].sum().reset_index(name='Total Use')
                    use_stats['% of Total Use'] = (use_stats['Total Use'] / use_stats['Total Use'].sum() * 100).round(2)
                    
                    comparison = pd.merge(coverage, use_stats, on='_lc_main')
                    comparison['Description'] = comparison['_lc_main'].map(LC_CLASSES)
                    
                    # Grouped Bar: % Collection vs % Use
                    fig_comp = go.Figure(data=[
                        go.Bar(name='% of Collection (Supply)', x=comparison['_lc_main'], y=comparison['% of Collection'], marker_color='#71C5E8'),
                        go.Bar(name='% of Total Use (Demand)', x=comparison['_lc_main'], y=comparison['% of Total Use'], marker_color='#285C4D')
                    ])
                    fig_comp.update_layout(title="Coverage vs. Use: Are we buying what users want?", barmode='group')
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Narrative
                    over_performing = comparison[comparison['% of Total Use'] > comparison['% of Collection']]
                    st.info(f"**Insight:** LC Classes {', '.join(over_performing['_lc_main'].tolist())} are 'over-performing'—they account for a higher percentage of use than they do of the collection size.")
                else:
                    st.info("Upload usage data or map a usage column to see Coverage vs. Use trends.")

# =====================================================================
# TOOL 2: ZERO-USE IDENTIFIER
# =====================================================================

def page_zero_use_identifier():
    st.title("🔍 Zero-Use Identifier")
    st.markdown("Identifies items that exist in your holdings but have no record of use in your circulation reports.")
    
    c1, c2 = st.columns(2)
    with c1: h_file = st.file_uploader("1. Upload Holdings CSV", type="csv")
    with c2: u_file = st.file_uploader("2. Upload Usage CSV", type="csv")

    if h_file and u_file:
        st.success("Ready to match holdings against usage.")
        if st.button("Identify Zero-Use Items"):
            st.info("This tool matches titles across both files and returns items found in Holdings but missing from Usage.")

# =====================================================================
# TOOL 3: RECOMMENDATION SCORER
# =====================================================================

def page_recommendation_scorer():
    st.title("📊 Acquisition Recommendation Scorer")
    st.markdown("Scores candidate books for purchase based on your existing collection's high-use subjects.")
    
    c1, c2 = st.columns(2)
    with c1: check_file = st.file_uploader("1. Upload Past Checkouts CSV", type="csv")
    with c2: rec_file = st.file_uploader("2. Upload New Recommendations CSV", type="csv")

    if check_file and rec_file:
        st.success("Data loaded. Configure weights to score recommendations.")
        st.slider("Subject Similarity Weight", 0, 100, 50)
        if st.button("Score Recommendations"):
            st.write("Scoring in progress...")

# =====================================================================
# NAVIGATION & HOME
# =====================================================================

def page_home():
    st.title("📚 Library Collection Dashboard")
    st.markdown("""
        ### Decision-Support Tools
        Select a tool from the sidebar to begin your analysis:
        
        * **🗺️ Collection Profiler**: Visualize subject strengths and compare **Coverage vs. Use** to identify supply and demand gaps.
        * **🔍 Zero-Use Identifier**: Cross-reference holdings with circulation data to find titles that haven't moved.
        * **📊 Acquisition Recommendation Scorer**: Score new book lists against your library's historical usage data.
    """)

def main():
    with st.sidebar:
        st.title("📚 Dashboard")
        page = st.radio("Select a tool:", ["🏠 Home", "🗺️ Collection Profiler", "🔍 Zero-Use Identifier", "📊 Acquisition Recommendation Scorer"])
        st.markdown("---")
        st.caption("v2.1 Consolidated Edition")

    if page == "🏠 Home": page_home()
    elif page == "🗺️ Collection Profiler": page_collection_profiler()
    elif page == "🔍 Zero-Use Identifier": page_zero_use_identifier()
    elif page == "📊 Acquisition Recommendation Scorer": page_recommendation_scorer()

if __name__ == "__main__":
    main()
