"""
Library Collection Dashboard v3.0
================================
Decision-Support Toolkit for Howard-Tilton Memorial Library

Tools Included:
  1. Collection Profiler — Visualize LC distribution & subject depth.
  2. Zero-Use Identifier — Detect low-ROI holdings.
  3. Acquisition Scorer — Evidence-based purchasing support.

Updated: 2025 | Contact: Kay P Maye (kmaye@tulane.edu)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from io import BytesIO

# =====================================================================
# CONFIG & THEMING
# =====================================================================

st.set_page_config(page_title="Library Collection Dashboard", page_icon="📚", layout="wide")

st.markdown("""
<style>
    :root { --tulane-green: #285C4D; --tulane-blue: #71C5E8; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: bold; font-size: 1.1rem; }
    .tool-header { color: #285C4D; border-bottom: 2px solid #285C4D; padding-bottom: 5px; margin-bottom: 15px; }
    .data-source-box { background-color: #f0f2f6; border-radius: 8px; padding: 15px; border-left: 5px solid #71C5E8; margin-bottom: 25px; }
    .usage-guide { background-color: #eef6f3; border-radius: 8px; padding: 15px; border-left: 5px solid #285C4D; margin-bottom: 10px; }
    .decision-box { background-color: #fff8e6; border-radius: 8px; padding: 15px; border-left: 5px solid #e6a800; margin-bottom: 20px; }
    .stButton>button { background-color: #285C4D; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HOME PAGE
# =====================================================================

def page_home():
    st.title("🏛️ Library Collection Decision Support Dashboard")
    st.markdown("""
    This dashboard consolidates three evidence-based tools to support collection development, 
    weeding, and strategic purchasing. Use the sidebar to navigate between tools.
    """)

    st.divider()

    # Tool summaries
    for icon, title, desc, decisions in [
        ("🗺️", "Collection Profiler", 
         "Analyze LC classification depth, subject distribution, and supply vs. demand patterns.",
         ["Where are we over/under collecting?", "Which LC areas show unmet demand?", "Which subjects need liaison attention?"]),
        
        ("🔍", "Zero-Use Identifier",
         "Identify holdings with no recorded use to support deselection or storage decisions.",
         ["Which items are candidates for off-site storage?", "Which areas show persistent non-use?", "What is the ROI of specific collections?"]),
        
        ("📊", "Acquisition Scorer",
         "Score candidate titles using local usage patterns, subject trends, and historical demand.",
         ["Which titles should we prioritize?", "How do vendor lists align with local needs?", "Where should end-of-year funds go?"])
    ]:
        st.markdown(f"<h3 class='tool-header'>{icon} {title}</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown(f"<div class='usage-guide'><strong>What it does:</strong><br>{desc}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='decision-box'><strong>Supports decisions such as:</strong><ul>" +
                        "".join([f"<li>{d}</li>" for d in decisions]) +
                        "</ul></div>", unsafe_allow_html=True)
        st.divider()

# =====================================================================
# TOOL 1: COLLECTION PROFILER
# =====================================================================

def page_collection_profiler():
    st.title("🗺️ Collection Profiler & Subject Analysis")

    st.markdown("""
    Use this tool to understand the *shape* of your collection: LC depth, subject coverage, 
    and how holdings align with actual usage. This helps identify:
    - Overbuilt areas with low demand  
    - Underbuilt areas with high demand  
    - Subjects needing liaison attention  
    - Gaps in accreditation or program support  
    """)

    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Display Settings")
        chart_height = st.slider("Chart Height", 400, 1400, 800)
        show_table = st.checkbox("Show Data Tables", True)

        st.header("📁 Data Files")
        st.caption("Upload catalog and optional usage files.")
        bib_file = st.file_uploader("Bibliographic / Catalog CSV", type="csv")
        use_file = st.file_uploader("Usage / Circulation CSV (Optional)", type="csv")

    if not bib_file:
        st.info("Upload a bibliographic file to begin.")
        return

    # Load data
    df_bib = pd.read_csv(bib_file)
    st.success(f"Loaded {len(df_bib):,} bibliographic records.")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🏛️ LC Distribution",
        "🏷️ Subject Analysis",
        "📈 Coverage vs Use"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("LC Classification Depth")

        st.markdown("""
        This visualization helps answer:
        - Which LC areas dominate the collection?
        - Where is the collection shallow or uneven?
        - Are there areas that may require rebalancing?
        """)

        fig = px.sunburst(
            df_bib,
            path=['Main_LC_Desc', 'Subclass_Desc'],
            height=chart_height,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_table:
            with st.expander("📋 LC Summary Table"):
                summary = df_bib.groupby('Main_LC_Desc').size().reset_index(name='Count')
                st.dataframe(summary, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("Subject Heading Concentration")

        st.markdown("""
        This view highlights the *intellectual shape* of the collection:
        - Which subjects dominate?
        - Are there emerging or neglected areas?
        - Do subject patterns align with academic programs?
        """)

        fig_sub = px.treemap(
            df_bib,
            path=['Subject_Heading'],
            height=chart_height,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("Supply vs. Demand (Coverage vs. Use)")

        st.markdown("""
        Compare what the library *owns* vs. what users *actually borrow*.  
        This helps identify:
        - Overbuilt areas with low demand  
        - High-demand areas needing investment  
        - Subjects where digital may outperform print  
        """)

        if not use_file:
            st.warning("Upload a usage file to enable this comparison.")
        else:
            df_use = pd.read_csv(use_file)

            # Placeholder comparison logic
            fig_gap = go.Figure(data=[
                go.Bar(name='Collection %', x=['A', 'B', 'C'], y=[30, 20, 50], marker_color='#71C5E8'),
                go.Bar(name='Usage %', x=['A', 'B', 'C'], y=[10, 40, 50], marker_color='#285C4D')
            ])
            fig_gap.update_layout(height=chart_height, barmode='group')
            st.plotly_chart(fig_gap, use_container_width=True)

# =====================================================================
# MAIN NAVIGATION
# =====================================================================

def main():
    with st.sidebar:
        st.title("📚 Dashboard Navigation")
        page = st.radio("Choose a Tool:", [
            "🏠 Home",
            "🗺️ Collection Profiler",
            "🔍 Zero-Use Identifier",
            "📊 Acquisition Scorer"
        ])
        st.divider()
        st.caption("Tulane University | Howard-Tilton Memorial Library")

    if page == "🏠 Home":
        page_home()
    elif page == "🗺️ Collection Profiler":
        page_collection_profiler()
    elif page == "🔍 Zero-Use Identifier":
        st.title("🔍 Zero-Use Identifier (Coming Soon)")
    elif page == "📊 Acquisition Scorer":
        st.title("📊 Acquisition Scorer (Coming Soon)")

if __name__ == "__main__":
    main()
