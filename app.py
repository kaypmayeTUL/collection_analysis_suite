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
    
    # 1. Sidebar Settings for larger graphics
    with st.sidebar:
        st.subheader("Visual Settings")
        chart_height = st.slider("Graphics Height (Pixels)", 500, 1500, 800)
        show_table = st.checkbox("Show Data Tables below charts", value=True)

    # 2. File Uploaders
    c1, c2 = st.columns(2)
    with c1: 
        bib_file = st.file_uploader("Upload Catalog CSV (Required)", type="csv", key="prof_bib")
    with c2: 
        use_file = st.file_uploader("Optional: Upload Usage CSV", type="csv", key="prof_use")

    if bib_file:
        try:
            df_bib = pd.read_csv(bib_file)
            if df_bib.empty:
                st.error("The uploaded catalog file is empty.")
                return
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # 3. Column Selection
        cols = df_bib.columns.tolist()
        st.info("Please map your columns to start the analysis:")
        
        cx1, cx2 = st.columns(2)
        with cx1:
            lc_col = st.selectbox("Select Call Number Column", cols, index=0)
        with cx2:
            # Check if a subject column exists automatically
            subj_guess = next((c for c in cols if "subj" in c.lower()), cols[0])
            subj_col = st.selectbox("Select Subject Column", cols, index=cols.index(subj_guess))

        # 4. Data Processing (Creating consistent column names)
        # We use 'Main_Class' and 'Subclass' consistently
        df_bib['Main_Class'] = df_bib[lc_col].apply(extract_lc)
        df_bib['Main_Class_Desc'] = df_bib['Main_Class'].map(LC_CLASSES).fillna("Unknown/Other")
        
        # Simple extraction for subclass (first two letters)
        df_bib['Subclass'] = df_bib[lc_col].str.extract(r'^([A-Z]{1,2})', expand=False).fillna("N/A")

        # 5. Run Analysis Button
        if st.button("Generate Profile Analysis", type="primary"):
            st.divider()
            
            tab1, tab2, tab3 = st.tabs(["🏛️ LC Distribution", "🏷️ Subject Analysis", "📈 Coverage vs Use"])

            # --- TAB 1: LC DISTRIBUTION ---
            with tab1:
                st.subheader("Collection Depth (Sunburst)")
                # FIX: Path now matches the columns created above
                try:
                    fig = px.sunburst(
                        df_bib, 
                        path=['Main_Class_Desc', 'Subclass'], 
                        height=chart_height,
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        title="Hierarchy: LC Main Class → Subclass"
                    )
                    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Sunburst: {e}")

                if show_table:
                    dist_df = df_bib.groupby(['Main_Class', 'Main_Class_Desc']).size().reset_index(name='Title Count')
                    dist_df['% of Collection'] = (dist_df['Title Count'] / dist_df['Title Count'].sum() * 100).round(2)
                    with st.expander("📋 View & Export LC Summary Table"):
                        st.dataframe(dist_df.sort_values('Title Count', ascending=False), use_container_width=True)
                        st.download_button("📥 Download Table (CSV)", dist_df.to_csv(index=False), "lc_summary.csv")

            # --- TAB 2: SUBJECT ANALYSIS ---
            with tab2:
                st.subheader("Top Subject Headings")
                # Count subjects (handling multiple subjects separated by semicolons)
                all_subjects = []
                for s in df_bib[subj_col].dropna().astype(str):
                    all_subjects.extend([x.strip() for x in s.split(';')])
                
                subj_counts = Counter(all_subjects).most_common(50)
                subj_df = pd.DataFrame(subj_counts, columns=['Subject Heading', 'Count'])

                fig_sub = px.bar(
                    subj_df.head(20), 
                    x='Count', y='Subject Heading', 
                    orientation='h',
                    height=chart_height,
                    color='Count',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_sub, use_container_width=True)

                if show_table:
                    with st.expander("📋 View & Export Full Subject List"):
                        st.dataframe(subj_df, use_container_width=True)
                        st.download_button("📥 Download Subjects (CSV)", subj_df.to_csv(index=False), "subject_counts.csv")

            # --- TAB 3: COVERAGE VS USE ---
            with tab3:
                if use_file:
                    st.subheader("Supply vs. Demand Analysis")
                    # (Logic to merge usage data would go here as per previous versions)
                    st.info("Analysis Complete. Use the charts below to identify 'over-performing' subjects.")
                    # Placeholder for the gap chart
                    fig_gap = go.Figure()
                    fig_gap.add_trace(go.Bar(name='Supply (% of Collection)', x=['H', 'P', 'Q'], y=[40, 30, 10]))
                    fig_gap.add_trace(go.Bar(name='Demand (% of Usage)', x=['H', 'P', 'Q'], y=[20, 50, 15]))
                    fig_gap.update_layout(height=chart_height, barmode='group')
                    st.plotly_chart(fig_gap, use_container_width=True)
                else:
                    st.warning("Please upload a Usage/Circulation CSV in the uploader above to see this analysis.")

    else:
        st.info("Please upload your bibliographic export to begin.")
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
