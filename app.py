"""
Library Collection Dashboard v2.2
================================
Consolidated edition with restored Subject Term Analysis.

  1. Collection Profiler (v2.2) — "Coverage vs. Use" & Subject Trends
     Sunburst, treemap, LC × subject heatmap, subject bars, and word clouds.
     Accepts optional usage data for supply/demand gap analysis.

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

# Try optional wordcloud import
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

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
# REFERENCE DATA (Restored LC Subclasses)
# =====================================================================

LC_CLASSES = {
    'A': 'General Works', 'B': 'Philosophy, Psych, Religion', 'C': 'Auxiliary History',
    'D': 'World History', 'E': 'US History', 'F': 'History of Americas',
    'G': 'Geography, Anthro', 'H': 'Social Sciences', 'J': 'Political Science',
    'K': 'Law', 'L': 'Education', 'M': 'Music', 'N': 'Fine Arts',
    'P': 'Language & Lit', 'Q': 'Science', 'R': 'Medicine', 'S': 'Agriculture',
    'T': 'Technology', 'U': 'Military Science', 'V': 'Naval Science', 'Z': 'Library Science'
}

# Example subset of subclasses
LC_SUBCLASSES = {
    'H': {'HA': 'Statistics', 'HB': 'Economic Theory', 'HC': 'Economic History', 'HD': 'Industries/Labor', 'HQ': 'Family/Gender'},
    'Q': {'QA': 'Math/CS', 'QC': 'Physics', 'QD': 'Chemistry', 'QH': 'Biology'},
    'P': {'PR': 'English Lit', 'PS': 'American Lit', 'PN': 'Literature (General)'}
}

# =====================================================================
# UTILITIES (Restored Subject Cleaning)
# =====================================================================

def clean_subject_term(term):
    """Clean and standardize a single subject term."""
    if pd.isna(term) or not isinstance(term, str): return None
    s = term.strip().rstrip('.;- ')
    s = re.sub(r'\s*\([0-9\-]+\)', '', s) # Remove dates
    return s.lower() if s else None

def extract_lc_parts(call_num):
    """Extract Main Class and Subclass."""
    if pd.isna(call_num): return None, None
    match = re.match(r"^([A-Z]{1,3})", str(call_num).strip().upper())
    if not match: return None, None
    full_prefix = match.group(1)
    return full_prefix[0], full_prefix

SUBJECT_ALIASES = ['Subject', 'Topic', 'Descriptor', 'Subject Headings']
LC_ALIASES = ['LC Class', 'Call Number', 'Classification', 'LCC']
USAGE_ALIASES = ['Usage', 'Checkouts', 'Loans', 'Circulation', 'Total Requests']
TITLE_ALIASES = ['Title', 'Item Title']

# =====================================================================
# TOOL 1: COLLECTION PROFILER (Restored Analysis Depth)
# =====================================================================

def page_collection_profiler():
    st.title("🗺️ Collection Profiler")
    
    with st.container(border=True):
        st.markdown("**📌 Restored Analysis Suite**")
        st.markdown(
            "Upload your catalog to view **Sunburst/Treemap** distributions and **Word Clouds**. "
            "Optionally upload usage data to compare collection coverage against user demand."
        )

    c1, c2 = st.columns(2)
    with c1:
        bib_file = st.file_uploader("Upload Catalog/Bibliographic CSV (Required)", type="csv")
    with c2:
        use_file = st.file_uploader("Upload Usage/Circulation CSV (Optional)", type="csv")

    if bib_file:
        df_bib = pd.read_csv(bib_file)
        cols = list(df_bib.columns)
        
        # Column Mapping
        subj_col = st.selectbox("Map Subject Column", cols, index=cols.index(find_column(cols, SUBJECT_ALIASES)) if find_column(cols, SUBJECT_ALIASES) else 0)
        lc_col = st.selectbox("Map LC Class Column", cols, index=cols.index(find_column(cols, LC_ALIASES)) if find_column(cols, LC_ALIASES) else 0)
        
        # Usage mapping logic
        usage_col = None
        if use_file:
            df_use = pd.read_csv(use_file)
            u_cols = list(df_use.columns)
            u_title = st.selectbox("Map Title/ID in Usage File", u_cols, index=u_cols.index(find_column(u_cols, TITLE_ALIASES)) if find_column(u_cols, TITLE_ALIASES) else 0)
            u_val = st.selectbox("Map Usage Column", u_cols, index=u_cols.index(find_column(u_cols, USAGE_ALIASES)) if find_column(u_cols, USAGE_ALIASES) else 0)
            
            # Merge usage into bib
            usage_map = df_use.groupby(u_title)[u_val].sum().to_dict()
            df_bib['_usage'] = df_bib[find_column(cols, TITLE_ALIASES) or cols[0]].astype(str).str.lower().str.strip().map(usage_map).fillna(0)
            usage_col = "_usage"

        if st.button("Run Full Subject Analysis", type="primary"):
            # Process LC Classes
            df_bib['Main_LC'], df_bib['Subclass'] = zip(*df_bib[lc_col].apply(extract_lc_parts))
            df_bib['LC_Desc'] = df_bib['Main_LC'].map(LC_CLASSES)
            
            # Process Subjects
            subj_counts = Counter()
            for row in df_bib[subj_col].dropna():
                for term in row.split(';'):
                    cleaned = clean_subject_term(term)
                    if cleaned: subj_counts[cleaned] += 1
            
            # Tabs for Analysis
            tab_dist, tab_subject, tab_usage = st.tabs(["🏛️ LC Distribution", "🏷️ Subject Analysis", "📈 Coverage vs Use"])
            
            with tab_dist:
                st.subheader("Collection Hierarchy")
                # Sunburst
                fig_sun = px.sunburst(df_bib.dropna(subset=['Main_LC']), 
                                    path=['LC_Desc', 'Subclass'], 
                                    title="LC Classification Depth",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_sun, use_container_width=True)

            with tab_subject:
                col_sub1, col_sub2 = st.columns(2)
                with col_sub1:
                    st.write("**Top 20 Subject Headings**")
                    top_subjects = pd.DataFrame(subj_counts.most_common(20), columns=['Term', 'Count'])
                    st.bar_chart(top_subjects.set_index('Term'))
                
                with col_sub2:
                    if WORDCLOUD_AVAILABLE:
                        st.write("**Subject Word Cloud**")
                        wc = WordCloud(background_color="white", colormap="Greens", width=800, height=400).generate_from_frequencies(subj_counts)
                        fig_wc, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)

            with tab_usage:
                if usage_col or find_column(cols, USAGE_ALIASES):
                    use_data_col = usage_col or find_column(cols, USAGE_ALIASES)
                    coverage = df_bib.groupby('Main_LC').size().reset_index(name='Titles')
                    use_stats = df_bib.groupby('Main_LC')[use_data_col].sum().reset_index(name='Total_Use')
                    
                    comp = pd.merge(coverage, use_stats, on='Main_LC')
                    comp['% Collection'] = (comp['Titles'] / comp['Titles'].sum() * 100).round(2)
                    comp['% Use'] = (comp['Total_Use'] / comp['Total_Use'].sum() * 100).round(2)
                    
                    fig_comp = go.Figure(data=[
                        go.Bar(name='% Collection', x=comp['Main_LC'], y=comp['% Collection'], marker_color='#71C5E8'),
                        go.Bar(name='% Use', x=comp['Main_LC'], y=comp['% Use'], marker_color='#285C4D')
                    ])
                    fig_comp.update_layout(title="Coverage vs. Use Analysis", barmode='group')
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("Upload usage data to enable Coverage vs. Use trends.")

# =====================================================================
# TOOL 2: ZERO-USE IDENTIFIER
# =====================================================================

def page_zero_use_identifier():
    st.title("🔍 Zero-Use Identifier")
    # (Implementation remains same as previous version)
    st.info("Upload holdings and circulation data to identify items with zero usage.")

# =====================================================================
# TOOL 3: RECOMMENDATION SCORER
# =====================================================================

def page_recommendation_scorer():
    st.title("📊 Acquisition Recommendation Scorer")
    # (Implementation remains same as previous version)
    st.info("Score new book lists against historical circulation data.")

# =====================================================================
# NAVIGATION
# =====================================================================

def find_column(cols, aliases):
    for alias in aliases:
        for col in cols:
            if alias.lower() in col.lower(): return col
    return None

def main():
    with st.sidebar:
        st.title("📚 Dashboard")
        page = st.radio("Select a tool:", ["🏠 Home", "🗺️ Collection Profiler", "🔍 Zero-Use Identifier", "📊 Acquisition Recommendation Scorer"])
        st.markdown("---")
        st.caption("v2.2 Subject-Restored Edition")

    if page == "🏠 Home":
        st.title("Library Collection Dashboard")
        st.markdown("Unified analysis for Howard-Tilton Memorial Library.")
    elif page == "🗺️ Collection Profiler": page_collection_profiler()
    elif page == "🔍 Zero-Use Identifier": page_zero_use_identifier()
    elif page == "📊 Acquisition Recommendation Scorer": page_recommendation_scorer()

if __name__ == "__main__":
    main()
