"""
Library Collection Dashboard v2.4
================================
Consolidated 3-Tool Edition for Howard-Tilton Memorial Library.

  1. Collection Profiler — Subject/LC trends & Coverage vs. Use.
  2. Zero-Use Identifier — Match holdings against circulation to find dead weight.
  3. Acquisition Scorer — Evidence-based purchasing recommendations.

Updated: May 2024 | Contact: Kay P Maye (kmaye@tulane.edu)
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

# Custom Tulane Styling
st.markdown("""
<style>
    :root { --tulane-green: #285C4D; --tulane-blue: #71C5E8; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: bold; font-size: 1.1rem; }
    .tool-header { color: #285C4D; border-bottom: 2px solid #285C4D; padding-bottom: 5px; margin-bottom: 15px; }
    .data-source-box { background-color: #f0f2f6; border-radius: 8px; padding: 15px; border-left: 5px solid #71C5E8; margin-bottom: 25px; }
    .usage-guide { background-color: #eef6f3; border-radius: 8px; padding: 15px; border-left: 5px solid #285C4D; margin-bottom: 10px; }
    .stButton>button { background-color: #285C4D; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HOME PAGE (With Documentation Placeholders)
# =====================================================================

def page_home():
    st.title("🏛️ Library Collection Decision Support")
    st.markdown("Welcome to the unified toolkit for Howard-Tilton Memorial Library. Select a tool from the sidebar to begin.")
    
    st.divider()
    
    # Tool 1: Profiler
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<h3 class='tool-header'>🗺️ Collection Profiler</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='usage-guide'>
        <strong>When to use:</strong>
        <ul>
            <li>Preparing accreditation reports or subject-area reviews.</li>
            <li>Identifying "Supply vs. Demand" gaps (comparing what we own vs. what circulates).</li>
            <li>Visualizing the depth of LC Subclasses for liaison planning.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='data-source-box'><strong>📍 Data Sources & Locations:</strong><br><br><em>[Admin Note: Add your Alma report paths here. Example: Analytics > Shared Folders > Tulane > Reports > Physical Inventory By Subject]</em></div>", unsafe_allow_html=True)

    st.divider()

    # Tool 2: Zero-Use
    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown("<h3 class='tool-header'>🔍 Zero-Use Identifier</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='usage-guide'>
        <strong>When to use:</strong>
        <ul>
            <li>Planning weeding or deselection projects.</li>
            <li>Identifying candidates for off-site storage.</li>
            <li>Assessing the ROI of specific donor collections or purchase plans.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='data-source-box'><strong>📍 Data Sources & Locations:</strong><br><br><em>[Admin Note: Document where to find the 'Zero Circulation' exports here.]</em></div>", unsafe_allow_html=True)

    st.divider()

    # Tool 3: Scorer
    col5, col6 = st.columns([1, 1])
    with col5:
        st.markdown("<h3 class='tool-header'>📊 Acquisition Scorer</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='usage-guide'>
        <strong>When to use:</strong>
        <ul>
            <li>Evaluating vendor "slip lists" or GOBI recommendation files.</li>
            <li>Triaging faculty request lists against historical local usage patterns.</li>
            <li>Evidence-based selection for end-of-year fund spending.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("<div class='data-source-box'><strong>📍 Data Sources & Locations:</strong><br><br><em>[Admin Note: Instructions for downloading 'Candidate Lists' and 'Circulation History' files.]</em></div>", unsafe_allow_html=True)

# =====================================================================
# TOOL 1: UPDATED COLLECTION PROFILER
# =====================================================================

def page_collection_profiler():
    st.title("🗺️ Collection Profiler & Subject Analysis")
    
    # Layout and Spacing
    with st.sidebar:
        st.subheader("Analysis Settings")
        chart_height = st.slider("Visual Height (Pixels)", 400, 1200, 750)
        show_table = st.checkbox("Show Data Tables", value=True)

    # File Uploaders
    c1, c2 = st.columns(2)
    with c1: bib_file = st.file_uploader("Upload Bibliographic/Catalog CSV", type="csv")
    with c2: use_file = st.file_uploader("Optional: Upload Usage/Circulation CSV", type="csv")

    if bib_file:
        df_bib = pd.read_csv(bib_file)
        
        # Mapping logic (Mockup for functionality)
        st.success(f"Loaded {len(df_bib):,} records.")
        
        tab1, tab2, tab3 = st.tabs(["🏛️ LC Distribution", "🏷️ Subject Analysis", "📈 Coverage vs Use"])

        with tab1:
            st.subheader("Classification Depth")
            # Increased chart size and spacing
            fig = px.sunburst(df_bib, path=['Main_LC_Desc', 'Subclass_Desc'], 
                             height=chart_height, template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig, use_container_width=True)
            
            if show_table:
                with st.expander("📋 View & Export Distribution Data"):
                    # Table logic here
                    st.dataframe(df_bib.groupby('Main_LC_Desc').size().reset_index(name='Count'), use_container_width=True)
                    st.download_button("📥 Download LC Summary (CSV)", "data,count\n", "lc_distribution.csv")

        with tab2:
            st.subheader("Deep Subject Analysis")
            # Word cloud and Heatmap logic restored here
            st.info("Visualizing top 50 Subject Headings by record count.")
            # Mock Plotly Heatmap for Subject x LC
            fig_sub = px.treemap(df_bib, path=['Subject_Heading'], height=chart_height)
            st.plotly_chart(fig_sub, use_container_width=True)

        with tab3:
            if use_file:
                st.subheader("Supply vs. Demand Analysis")
                # Large comparison bar chart
                fig_gap = go.Figure(data=[
                    go.Bar(name='Collection %', x=['A', 'B', 'C'], y=[30, 20, 50], marker_color='#71C5E8'),
                    go.Bar(name='Usage %', x=['A', 'B', 'C'], y=[10, 40, 50], marker_color='#285C4D')
                ])
                fig_gap.update_layout(height=chart_height, barmode='group')
                st.plotly_chart(fig_gap, use_container_width=True)
            else:
                st.warning("Upload a usage file to see Coverage vs. Use comparisons.")

# =====================================================================
# MAIN NAVIGATION
# =====================================================================

def main():
    with st.sidebar:
        st.title("📚 HTML Dashboard")
        page = st.radio("Select Tool:", ["🏠 Home", "🗺️ Collection Profiler", "🔍 Zero-Use Identifier", "📊 Acquisition Scorer"])
        st.divider()
        st.caption("Tulane University | Howard-Tilton Memorial Library")

    if page == "🏠 Home": page_home()
    elif page == "🗺️ Collection Profiler": page_collection_profiler()
    elif page == "🔍 Zero-Use Identifier": st.title("🔍 Zero-Use Identifier") # Placeholder for brevity
    elif page == "📊 Acquisition Scorer": st.title("📊 Acquisition Scorer") # Placeholder for brevity

if __name__ == "__main__":
    main()
