"""
Library Collection Dashboard
============================
A unified Streamlit application combining three collection decision-support tools:

  1. Collection Profiler — "What does our collection look like?"
     Sunburst, treemap, LC × subject heatmap, subject bars, word cloud, gap analysis.
     Feeds collection assessment, accreditation, weeding prep, liaison planning.

  2. Usage & Subscription Analyzer — "What's being used and what isn't?"
     Title-level usage analysis for COUNTER 5 e-resource reports and print
     circulation data. Top titles, cancellation review, publisher rollups,
     monthly trends. Feeds renewal, cancellation, and weeding decisions.

  3. Acquisition Recommendation Scorer — "What should we buy next?"
     Scores new candidate books against checkout history using subject
     similarity, LC fit, author popularity, and faculty research interests.
     Feeds purchasing, approval-plan review, and faculty request triage.

v2.0 — Consolidated edition
Contact: Kay P Maye (kmaye@tulane.edu)
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
from difflib import SequenceMatcher

# Conditional imports
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import nltk
    from nltk.stem import SnowballStemmer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# =====================================================================
# PAGE CONFIG & GLOBAL CSS (Tulane palette)
# =====================================================================

st.set_page_config(
    page_title="Library Collection Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --tulane-green: #285C4D;
    --tulane-blue: #71C5E8;
}
.main > div { padding-top: 1.5rem; }
.stButton>button {
    background-color: #285C4D;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    border: none;
    width: 100%;
}
.stButton>button:hover { background-color: #1e4a3c; }
div[data-testid="metric-container"] {
    background-color: #eef6f3;
    border: 1px solid #285C4D;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.uploadbox {
    border: 2px dashed #285C4D;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    background-color: #eef6f3;
}
.decision-box {
    background-color: #eef6f3;
    border-left: 4px solid #285C4D;
    padding: 15px 20px;
    border-radius: 4px;
    margin: 10px 0;
}
.tool-card {
    background-color: #f8faf9;
    border: 1px solid #d4e4df;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SHARED: LC Classification reference
# =====================================================================

LC_CLASSES = {
    'A': 'General Works', 'B': 'Philosophy, Psychology, Religion',
    'C': 'Auxiliary Sciences of History', 'D': 'World History',
    'E': 'US History', 'F': 'History of the Americas',
    'G': 'Geography, Anthropology, Recreation', 'H': 'Social Sciences',
    'J': 'Political Science', 'K': 'Law', 'L': 'Education',
    'M': 'Music & Books on Music', 'N': 'Fine Arts', 'P': 'Language & Literature',
    'Q': 'Science', 'R': 'Medicine', 'S': 'Agriculture',
    'T': 'Technology', 'U': 'Military Science', 'V': 'Naval Science',
    'Z': 'Bibliography & Library Science'
}

LC_SUBCLASSES = {
    'B': {
        'B': 'Philosophy (General)', 'BC': 'Logic', 'BD': 'Speculative Philosophy',
        'BF': 'Psychology', 'BJ': 'Ethics', 'BL': 'Religions, Mythology',
        'BM': 'Judaism', 'BP': 'Islam, Bahai', 'BQ': 'Buddhism',
        'BR': 'Christianity', 'BS': 'The Bible', 'BT': 'Doctrinal Theology',
        'BV': 'Practical Theology', 'BX': 'Christian Denominations'
    },
    'H': {
        'H': 'Social Sciences (General)', 'HA': 'Statistics',
        'HB': 'Economic Theory', 'HC': 'Economic History & Conditions',
        'HD': 'Industries, Land Use, Labor', 'HE': 'Transportation & Communications',
        'HF': 'Commerce', 'HG': 'Finance', 'HJ': 'Public Finance',
        'HM': 'Sociology', 'HN': 'Social History & Conditions',
        'HQ': 'The Family, Marriage, Gender', 'HS': 'Societies & Clubs',
        'HT': 'Communities, Classes, Races', 'HV': 'Social Pathology & Welfare',
        'HX': 'Socialism, Communism, Anarchism'
    },
    'P': {
        'P': 'Philology & Linguistics (General)', 'PA': 'Greek & Latin',
        'PB': 'Modern Languages (General)', 'PC': 'Romance Languages',
        'PD': 'Germanic Languages', 'PE': 'English',
        'PF': 'West Germanic Languages', 'PG': 'Slavic & Baltic Languages',
        'PH': 'Uralic & Basque Languages', 'PJ': 'Semitic Languages',
        'PK': 'Indo-Iranian Languages', 'PL': 'East Asian & Oceanic Languages',
        'PM': 'Indigenous Languages of the Americas',
        'PN': 'Literature (General)', 'PQ': 'French, Italian, Spanish, Portuguese Literature',
        'PR': 'English Literature', 'PS': 'American Literature',
        'PT': 'German, Dutch, Scandinavian Literature', 'PZ': 'Fiction & Juvenile Literature'
    },
    'Q': {
        'Q': 'Science (General)', 'QA': 'Mathematics',
        'QB': 'Astronomy', 'QC': 'Physics', 'QD': 'Chemistry',
        'QE': 'Geology', 'QH': 'Natural History & Biology',
        'QK': 'Botany', 'QL': 'Zoology', 'QM': 'Human Anatomy',
        'QP': 'Physiology', 'QR': 'Microbiology'
    },
    'R': {
        'R': 'Medicine (General)', 'RA': 'Public Health',
        'RB': 'Pathology', 'RC': 'Internal Medicine',
        'RD': 'Surgery', 'RE': 'Ophthalmology', 'RF': 'Otorhinolaryngology',
        'RG': 'Gynecology & Obstetrics', 'RJ': 'Pediatrics',
        'RK': 'Dentistry', 'RL': 'Dermatology', 'RM': 'Therapeutics & Pharmacology',
        'RS': 'Pharmacy', 'RT': 'Nursing', 'RZ': 'Alternative Medicine'
    },
    'J': {
        'J': 'General Legislative & Executive', 'JA': 'Political Science (General)',
        'JC': 'Political Theory', 'JF': 'Political Institutions (General)',
        'JK': 'US Political Institutions', 'JL': 'Latin American Political Institutions',
        'JN': 'European Political Institutions', 'JQ': 'Asian/African Political Institutions',
        'JS': 'Local Government', 'JV': 'Colonies & Colonization',
        'JX': 'International Law (old)', 'JZ': 'International Relations'
    },
    'K': {
        'K': 'Law (General)', 'KB': 'Religious Law',
        'KD': 'Law of the UK & Ireland', 'KE': 'Law of Canada',
        'KF': 'Law of the US', 'KG': 'Law of Latin America',
        'KJ': 'Law of Europe', 'KZ': 'Law of Nations'
    },
    'T': {
        'T': 'Technology (General)', 'TA': 'Engineering (General)',
        'TC': 'Hydraulic & Ocean Engineering', 'TD': 'Environmental Technology',
        'TE': 'Highway Engineering', 'TF': 'Railroad Engineering',
        'TG': 'Bridge Engineering', 'TH': 'Building Construction',
        'TJ': 'Mechanical Engineering', 'TK': 'Electrical & Nuclear Engineering',
        'TL': 'Motor Vehicles, Aeronautics', 'TN': 'Mining & Metallurgy',
        'TP': 'Chemical Technology', 'TR': 'Photography',
        'TS': 'Manufactures', 'TT': 'Handicrafts & Arts',
        'TX': 'Home Economics'
    }
}

LC_NAMES_EXTENDED = {
    "A": "General Works", "AC": "Collections, Series, Collected Works",
    "AE": "Encyclopedias", "AG": "Dictionaries and Other General Reference Works",
    "AI": "Indexes", "AM": "Museums, Collectors and Collecting",
    "AN": "Newspapers", "AP": "Periodicals",
    "AS": "Academies and Learned Societies", "AY": "Yearbooks, Almanacs, Directories",
    "AZ": "History of Scholarship and Learning",
    "B": "Philosophy (General)", "BC": "Logic", "BD": "Speculative Philosophy",
    "BF": "Psychology", "BH": "Aesthetics", "BJ": "Ethics",
    "BL": "Religion (General), Mythology, Rationalism",
    "BM": "Judaism", "BP": "Islam, Bahaism, Theosophy", "BQ": "Buddhism",
    "BR": "Christianity", "BS": "The Bible", "BT": "Doctrinal Theology",
    "BV": "Practical Theology", "BX": "Christian Denominations",
    "C": "Auxiliary Sciences of History (General)", "CB": "History of Civilization",
    "CC": "Archaeology", "CD": "Diplomatics, Archives, Seals",
    "CE": "Technical Chronology, Calendar", "CJ": "Numismatics",
    "CN": "Inscriptions, Epigraphy", "CR": "Heraldry",
    "CS": "Genealogy", "CT": "Biography",
    "D": "World History (General)", "DA": "History of Great Britain",
    "DB": "History of Austria, Liechtenstein, Hungary, Czechia, Slovakia",
    "DC": "History of France, Andorra, Monaco", "DD": "History of Germany",
    "DE": "History of the Greco-Roman World", "DF": "History of Greece",
    "DG": "History of Italy, Vatican City, San Marino, Malta",
    "DH": "History of Low Countries, Benelux", "DJ": "History of Netherlands",
    "DK": "History of Russia, Soviet Union, Former Soviet Republics",
    "DL": "History of Northern Europe, Scandinavia",
    "DP": "History of Iberian Peninsula, Spain, Portugal",
    "DQ": "History of Switzerland", "DR": "History of Balkan Peninsula",
    "DS": "History of Asia", "DT": "History of Africa",
    "DU": "History of Oceania, Pacific Area", "DX": "History of Romani People",
    "E": "American History (General), United States History",
    "F": "Local History of the United States and British, Dutch, French, Latin America",
    "G": "Geography (General), Atlases, Maps",
    "GA": "Mathematical Geography, Cartography", "GB": "Physical Geography",
    "GC": "Oceanography", "GE": "Environmental Sciences",
    "GF": "Human Ecology, Anthropogeography", "GN": "Anthropology",
    "GR": "Folklore", "GT": "Manners and Customs", "GV": "Recreation, Leisure",
    "H": "Social Sciences (General)", "HA": "Statistics",
    "HB": "Economic Theory, Demography", "HC": "Economic History and Conditions",
    "HD": "Industries, Land Use, Labor", "HE": "Transportation and Communications",
    "HF": "Commerce", "HG": "Finance", "HJ": "Public Finance",
    "HM": "Sociology (General)", "HN": "Social History and Conditions, Social Problems, Social Reform",
    "HQ": "The Family, Marriage, Women, Sexuality",
    "HT": "Communities, Classes, Races",
    "HV": "Social Pathology, Criminology, Public Welfare",
    "HX": "Socialism, Communism, Anarchism",
    "J": "Political Science (General)",
    "JC": "Political Theory",
    "JF": "Political Institutions and Public Administration (General)",
    "JK": "Political Institutions — United States",
    "JL": "Political Institutions — Canada, Latin America",
    "JN": "Political Institutions — Europe",
    "JQ": "Political Institutions — Asia, Africa, Pacific Area",
    "JS": "Local Government, Municipal Government",
    "JV": "Colonies and Colonization, Emigration and Immigration",
    "JZ": "International Relations",
    "K": "Law (General)", "KD": "Law of the United Kingdom and Ireland",
    "KE": "Law of Canada", "KF": "Law of the United States",
    "KZ": "Law of Nations, International Law",
    "L": "Education (General)", "LA": "History of Education",
    "LB": "Theory and Practice of Education", "LC": "Social Aspects of Education",
    "M": "Music", "ML": "Literature on Music", "MT": "Musical Instruction and Study",
    "N": "Visual Arts (General)", "NA": "Architecture", "NB": "Sculpture",
    "NC": "Drawing, Design, Illustration", "ND": "Painting",
    "NE": "Print Media", "NK": "Decorative Arts", "NX": "Arts in General",
    "P": "Philology and Linguistics (General)",
    "PA": "Greek and Latin Language and Literature",
    "PN": "Literature (General)",
    "PQ": "French, Italian, Spanish, Portuguese Literatures",
    "PR": "English Literature", "PS": "American Literature",
    "PZ": "Fiction and Juvenile Literature",
    "Q": "Science (General)", "QA": "Mathematics", "QB": "Astronomy",
    "QC": "Physics", "QD": "Chemistry", "QE": "Geology",
    "QH": "Natural History, Biology (General)", "QK": "Botany",
    "QL": "Zoology", "QM": "Human Anatomy", "QP": "Physiology",
    "QR": "Microbiology",
    "R": "Medicine (General)", "RA": "Public Aspects of Medicine",
    "RB": "Pathology", "RC": "Internal Medicine", "RD": "Surgery",
    "RM": "Therapeutics, Pharmacology", "RS": "Pharmacy and Materia Medica",
    "RT": "Nursing",
    "S": "Agriculture (General)", "SB": "Plant Culture, Horticulture",
    "SD": "Forestry", "SF": "Animal Culture, Veterinary Medicine",
    "T": "Technology (General)", "TA": "Engineering (General), Civil Engineering",
    "TK": "Electrical Engineering, Electronics, Nuclear Engineering",
    "TL": "Motor Vehicles, Aeronautics, Astronautics",
    "U": "Military Science (General)", "V": "Naval Science (General)",
    "VM": "Naval Architecture, Shipbuilding, Marine Engineering",
    "Z": "Bibliography, Library Science, Information Resources (General)",
    "ZA": "Information Resources, Information Science",
}


# =====================================================================
# SHARED: Text normalization & LC utilities
# =====================================================================

_RE_DATE_PAREN = re.compile(r'\s*\([0-9\-]+\)')
_RE_MULTI_SPACE = re.compile(r'\s+')
_RE_DASH_SPACE = re.compile(r'\s*-\s*')


def normalize_text(text):
    """Lowercase → strip accents → clean punctuation → collapse whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_subject_term(term):
    """Clean and standardize a single subject term."""
    if pd.isna(term) or not isinstance(term, str) or term.strip() == '':
        return None
    s = term.strip().rstrip('.;- ')
    s = _RE_DATE_PAREN.sub('', s)
    s = s.replace('--', ' - ')
    s = _RE_DASH_SPACE.sub(' - ', s)
    s = _RE_MULTI_SPACE.sub(' ', s).strip()
    return s.lower() if s else None


def split_subjects(raw_subjects):
    """Split on ; | , newline and normalize each piece."""
    if pd.isna(raw_subjects) or not isinstance(raw_subjects, str):
        return []
    parts = re.split(r"[;|,\n]", raw_subjects)
    return [normalize_text(p) for p in parts if normalize_text(p)]


def extract_lc_prefix(lc_class):
    """Extract LC letter prefix from a call number string."""
    if pd.isna(lc_class):
        return None
    match = re.match(r"^([A-Z]{1,3})", str(lc_class).strip().upper())
    return match.group(1) if match else None


def _extract_lc_vectorized(series):
    """Vectorized LC class extraction — returns (main_class, subclass) Series."""
    cleaned = series.astype(str).str.strip().str.upper()
    letters = cleaned.str.extract(r'^([A-Z]{1,3})', expand=False)
    main_class = letters.str[0]
    mask = series.isna() | (series.astype(str).str.strip() == '')
    main_class = main_class.where(~mask, other=None)
    letters = letters.where(~mask, other=None)
    return main_class, letters


# =====================================================================
# SHARED: Column detection & CSV loading
# =====================================================================

SUBJECT_ALIASES = ['Subjects', 'Subject', 'Subject Terms', 'Subject Headings',
                   'SUBJECT', 'subject_terms', 'Topics']
LC_ALIASES = ['LC Classification Code', 'LC Classification', 'LC Class',
              'Call Number', 'CallNumber', 'call_number', 'Call #',
              'LC Call Number', 'LCC', 'Classification', 'lc_classification']
TITLE_ALIASES = ['Title', 'title', 'TITLE', 'Book Title', 'Item Title']
WEIGHT_ALIASES = ['Loans', 'Loans (Total)', 'Checkouts', 'Circulation',
                  'Total_Item_Requests', 'Usage', 'Count',
                  'Digital File Downloads', 'Digital File Views', 'checkouts']


def find_column(df_or_cols, aliases, partial=True):
    """Find a column matching any alias. Accepts a DataFrame or list of column names."""
    cols = list(df_or_cols.columns) if isinstance(df_or_cols, pd.DataFrame) else list(df_or_cols)
    for alias in aliases:
        if alias in cols:
            return alias
        if partial:
            for col in cols:
                if alias.lower() in col.lower():
                    return col
    return None


def _detect_columns_from_header(file_bytes):
    """Read only the header row to detect column names without loading all data."""
    try:
        header = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', nrows=0)
    except Exception:
        header = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', nrows=0)
    return [c.strip() for c in header.columns]


@st.cache_data(show_spinner=False)
def _load_csv_chunked(file_bytes, filename, cols_to_keep=None):
    """Load CSV with minimal memory footprint."""
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False,
                         usecols=cols_to_keep)
    except Exception:
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', low_memory=False,
                             usecols=cols_to_keep)
        except Exception:
            try:
                df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False)
            except Exception:
                df = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# =====================================================================
# SHARED: Footer & decision-box helper
# =====================================================================

def _footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Library Collection Dashboard v2.0 | Howard-Tilton Memorial Library, Tulane University</p>
        <p>For support, contact Kay P Maye at kmaye@tulane.edu</p>
    </div>
    """, unsafe_allow_html=True)


def _decision_box(title, body_md):
    """Render a styled 'when to use this' callout using native Streamlit.

    Uses st.container with a border and markdown inside, which renders reliably
    regardless of indentation quirks in HTML-in-markdown.
    """
    with st.container(border=True):
        st.markdown(f"**📌 {title}**")
        st.markdown(body_md)


# =====================================================================
# =====================================================================
# TOOL 1: COLLECTION PROFILER
# =====================================================================
# "What does our collection look like?"
# Merged from: Collection Strengths Tool + Use Analysis Tool (word cloud)
# =====================================================================
# =====================================================================

CHUNK_SIZE = 50_000


def _profiler_process_subjects_chunk(subj_series, weight_series, lc_series,
                                     subject_counter, subject_by_lc):
    """Build subject frequency tables in one pass, optionally broken down by LC."""
    subj_arr = subj_series.values
    weight_arr = weight_series.values
    lc_arr = lc_series.values
    for i in range(len(subj_arr)):
        raw = subj_arr[i]
        if not isinstance(raw, str) or raw == '':
            continue
        w = weight_arr[i]
        lc = lc_arr[i]
        parts = raw.split(';')
        for part in parts:
            cleaned = clean_subject_term(part)
            if cleaned:
                subject_counter[cleaned] += w
                # Also count subclass components for deeper word cloud richness
                if ' - ' in cleaned:
                    for comp in cleaned.split(' - '):
                        comp = comp.strip()
                        if comp:
                            subject_counter[comp] += w
                if lc is not None and isinstance(lc, str):
                    subject_by_lc[lc][cleaned] += w


def _profiler_run_analysis(df, subj_col, lc_col, title_col, weight_col,
                           selected_classes, progress_bar):
    """Single pass that builds everything: LC counts, subject counter, subject-by-LC, gaps."""
    n_total = len(df)
    if selected_classes is not None and lc_col:
        mask = df['_lc_main'].isin(selected_classes) | df['_lc_main'].isna()
        idx = df.index[mask]
    else:
        idx = df.index
    n_records = len(idx)
    if weight_col:
        weight_all = df.loc[idx, weight_col]
    else:
        weight_all = pd.Series(1.0, index=idx)
    total_weight = weight_all.sum()
    results = {'n_records': n_records, 'total_weight': total_weight}
    progress_bar.progress(5, "Analyzing LC classifications...")

    if lc_col:
        lc_main = df.loc[idx, '_lc_main'].dropna()
        lc_sub = df.loc[idx, '_lc_sub'].dropna()
        lc_main_w = weight_all.reindex(lc_main.index)
        lc_sub_w = weight_all.reindex(lc_sub.index)
        lc_main_counts = lc_main_w.groupby(lc_main).sum().to_dict()
        lc_sub_counts = lc_sub_w.groupby(lc_sub).sum().to_dict()

        sunburst_rows = []
        for mc, mcount in sorted(lc_main_counts.items(), key=lambda x: -x[1]):
            label = LC_CLASSES.get(mc, mc)
            sunburst_rows.append({'id': mc, 'parent': '', 'label': f"{mc} – {label}", 'value': mcount})
            sub_dict = LC_SUBCLASSES.get(mc, {})
            for sc, scount in sorted(lc_sub_counts.items(), key=lambda x: -x[1]):
                if sc.startswith(mc) and sc != mc:
                    sl = sub_dict.get(sc, sc)
                    sunburst_rows.append({'id': sc, 'parent': mc, 'label': f"{sc} – {sl}", 'value': scount})
        results['lc_main_counts'] = lc_main_counts
        results['lc_sub_counts'] = lc_sub_counts
        results['sunburst_data'] = sunburst_rows
    else:
        results['lc_main_counts'] = {}
        results['lc_sub_counts'] = {}
        results['sunburst_data'] = []

    progress_bar.progress(15, "Processing subject terms...")

    if subj_col:
        subject_counter = Counter()
        subject_by_lc = defaultdict(Counter)
        subj_full = df.loc[idx, subj_col]
        lc_full = df.loc[idx, '_lc_main'] if lc_col else pd.Series(None, index=idx)
        n_chunks = (n_records // CHUNK_SIZE) + 1
        for ci in range(n_chunks):
            start = ci * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_records)
            if start >= n_records:
                break
            cidx = idx[start:end]
            _profiler_process_subjects_chunk(
                subj_full.loc[cidx], weight_all.loc[cidx], lc_full.loc[cidx],
                subject_counter, subject_by_lc
            )
            pct = 15 + int(65 * (end / n_records))
            progress_bar.progress(pct, f"Processed {end:,} of {n_records:,} records...")
        results['subject_counts'] = subject_counter
        results['subject_by_lc'] = dict(subject_by_lc)
    else:
        results['subject_counts'] = Counter()
        results['subject_by_lc'] = {}

    progress_bar.progress(85, "Running gap analysis...")

    if lc_col:
        all_classes = set(LC_CLASSES.keys())
        present = set(results['lc_main_counts'].keys())
        results['missing_lc_classes'] = sorted(all_classes - present)
        thin_t = total_weight * 0.01 if total_weight > 0 else 0
        results['thin_lc_classes'] = {c: v for c, v in results['lc_main_counts'].items() if v < thin_t}
    else:
        results['missing_lc_classes'] = []
        results['thin_lc_classes'] = {}

    results['detail_available'] = bool(title_col or lc_col or subj_col)
    results['detail_cols'] = [c for c in [title_col, lc_col, subj_col] if c]
    results['detail_total'] = n_records
    progress_bar.progress(100, "Done!")
    gc.collect()
    return results


def _profiler_display_results(results, settings, df, idx):
    """Render all enabled visualizations in a sensible narrative order."""
    wl = settings['weight_label']
    top_n = settings['top_n_subjects']

    st.markdown("---")
    st.subheader("📊 Collection Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records Analyzed", f"{results['n_records']:,}")
    c2.metric(f"Total {wl}", f"{results['total_weight']:,.0f}")
    c3.metric("LC Classes Present", f"{len(results['lc_main_counts'])}")
    c4.metric("Unique Subjects", f"{len(results['subject_counts']):,}")

    # Word cloud (formerly its own tool — now a view option here)
    if settings['show_wordcloud'] and results['subject_counts']:
        st.markdown("---")
        st.subheader(f"☁️ Subject Word Cloud ({wl}-weighted)")
        if not WORDCLOUD_AVAILABLE:
            st.warning("Install `wordcloud` and `matplotlib` to enable this view: "
                       "`pip install wordcloud matplotlib`")
        else:
            min_len = settings.get('wc_min_len', 3)
            max_words = settings.get('wc_max_words', 100)
            color_scheme = settings.get('wc_color', 'viridis')
            filtered = {t: c for t, c in results['subject_counts'].items()
                        if t and len(t) >= min_len}
            if filtered:
                wc = WordCloud(
                    width=1200, height=500, background_color='white',
                    colormap=color_scheme, max_words=max_words,
                    relative_scaling=0.5, min_font_size=10, prefer_horizontal=0.7,
                ).generate_from_frequencies(filtered)
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                # Save word cloud image for download
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                buf.seek(0)
                st.download_button("📥 Download Word Cloud (PNG)", buf,
                                   "collection_wordcloud.png", "image/png",
                                   key='prof_dl_wc')
                plt.close(fig)
            else:
                st.info(f"No subjects meet the minimum length of {min_len} characters.")

    if settings['show_sunburst'] and results['sunburst_data']:
        st.markdown("---")
        st.subheader("🌞 LC Classification Sunburst")
        sb = results['sunburst_data']
        fig = go.Figure(go.Sunburst(
            ids=[r['id'] for r in sb], labels=[r['label'] for r in sb],
            parents=[r['parent'] for r in sb], values=[r['value'] for r in sb],
            branchvalues='total',
            marker=dict(colors=[r['value'] for r in sb],
                        colorscale=[[0, '#71C5E8'], [0.5, '#285C4D'], [1, '#1a3d33']]),
            maxdepth=2
        ))
        fig.update_layout(height=600, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if settings['show_treemap'] and results['lc_main_counts']:
        st.markdown("---")
        st.subheader("🗺️ LC Classification Treemap")
        tm_data = [{'Class': f"{c} – {LC_CLASSES.get(c, c)}", 'Count': ct,
                    'Pct': ct / results['total_weight'] * 100}
                   for c, ct in sorted(results['lc_main_counts'].items(), key=lambda x: -x[1])]
        tm_df = pd.DataFrame(tm_data)
        fig = px.treemap(tm_df, path=['Class'], values='Count', color='Count',
                         color_continuous_scale=[[0, '#71C5E8'], [0.5, '#285C4D'], [1, '#1a3d33']],
                         hover_data={'Pct': ':.1f'})
        fig.update_layout(height=500, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if settings['show_subject_bars'] and results['subject_counts']:
        st.markdown("---")
        st.subheader(f"📖 Top {top_n} Subject Terms")
        top_subjects = results['subject_counts'].most_common(top_n)
        sdf = pd.DataFrame(top_subjects, columns=['Subject', 'Count'])
        fig = px.bar(sdf, x='Count', y='Subject', orientation='h', color='Count',
                     color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          height=max(450, top_n * 24), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("📥 Download Subject Frequencies (CSV)",
                           sdf.to_csv(index=False), "subject_frequencies.csv",
                           "text/csv", key='prof_dl_subj')

    if settings['show_heatmap'] and results['subject_by_lc']:
        st.markdown("---")
        st.subheader("🔥 LC Class × Top Subjects Heatmap")
        global_top = [s for s, _ in results['subject_counts'].most_common(min(top_n, 25))]
        lc_present = sorted(results['subject_by_lc'].keys())
        matrix = [[results['subject_by_lc'][c].get(s, 0) for s in global_top] for c in lc_present]
        labels_y = [f"{c} – {LC_CLASSES.get(c, c)}" for c in lc_present]
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=global_top, y=labels_y,
            colorscale=[[0, '#ffffff'], [0.3, '#71C5E8'], [1, '#285C4D']],
        ))
        fig.update_layout(height=max(400, len(lc_present) * 35), xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

    if settings['show_gap_analysis']:
        st.markdown("---")
        st.subheader("🔎 Collection Gap Analysis")
        missing = results['missing_lc_classes']
        thin = results['thin_lc_classes']
        if missing:
            st.markdown("**LC Classes with No Holdings:**")
            st.dataframe(pd.DataFrame([{'LC Class': c, 'Description': LC_CLASSES.get(c, '')}
                                       for c in missing]), use_container_width=True, hide_index=True)
        else:
            st.info("All LC main classes are represented.")
        if thin:
            st.markdown(f"**LC Classes Below 1% of Collection:**")
            rows_t = [{'LC Class': c, 'Description': LC_CLASSES.get(c, ''),
                       f'{wl}': f"{v:,.0f}",
                       '% of Collection': f"{v / results['total_weight'] * 100:.2f}%"}
                      for c, v in sorted(thin.items(), key=lambda x: x[1])]
            st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)
        if results['lc_main_counts']:
            st.markdown("**Strongest Areas (top 5):**")
            top5 = sorted(results['lc_main_counts'].items(), key=lambda x: -x[1])[:5]
            rows_s = [{'LC Class': c, 'Description': LC_CLASSES.get(c, ''),
                       f'{wl}': f"{v:,.0f}",
                       '% of Collection': f"{v / results['total_weight'] * 100:.1f}%"}
                      for c, v in top5]
            st.dataframe(pd.DataFrame(rows_s), use_container_width=True, hide_index=True)

    if settings['show_detail_table'] and results.get('detail_available') and df is not None:
        st.markdown("---")
        st.subheader("📋 Title Details (paginated)")
        PAGE_SIZE = 5_000
        total = results['detail_total']
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = st.number_input("Page", 1, total_pages, 1, key='prof_page')
        start = (page - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        st.caption(f"Records {start + 1:,}–{end:,} of {total:,}")
        page_idx = idx[start:end]
        st.dataframe(df.loc[page_idx, results['detail_cols']],
                     use_container_width=True, height=400, hide_index=True)


def page_collection_profiler():
    """Tool 1: Collection Profiler."""
    if 'prof_results' not in st.session_state:
        st.session_state['prof_results'] = None
    if 'prof_filtered_idx' not in st.session_state:
        st.session_state['prof_filtered_idx'] = None

    st.header("🗺️ Collection Profiler")
    st.markdown(
        "**What does our collection look like?** Upload a title list with **subject terms** "
        "and/or **LC call numbers** to map disciplinary strengths, spot gaps, and explore "
        "subject coverage."
    )
    _decision_box(
        "When to use this tool",
        "- **Collections:** Baseline assessment, accreditation self-studies, weeding prep "
        "(find thin/missing areas), justifying budget asks by showing strengths.\n"
        "- **Instruction:** Prepare for a liaison session — see at a glance what you "
        "actually have in HQ or PN before walking into the class.\n"
        "- **Outreach:** Quick visuals for faculty meetings and annual reports "
        "(\"here's what sociology looks like in our collection\")."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="uploadbox">
            <h2>📂 Upload Title List (CSV)</h2>
            <p>Drag and drop or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose CSV", type=['csv'],
            help="Needs Subjects and/or LC Classification; usage columns are optional.",
            label_visibility="collapsed", key="prof_upload"
        )

    if uploaded_file is None:
        with st.expander("📖 How to use this tool", expanded=True):
            st.markdown("""
            **Step 1:** Prepare a CSV with some combination of `Subjects`, `LC Classification` /
            `Call Number`, `Title`, and optionally a usage column like `Loans` or `Checkouts`.

            **Step 2:** Upload and configure analysis settings in the sidebar.

            **Step 3:** Click **Analyze Collection** to see sunburst charts, treemaps,
            top subjects, word cloud, heatmaps, and gap analysis.

            Optimized for datasets of 500K–1M+ records.
            """)
        return

    file_bytes = uploaded_file.getvalue()
    all_cols = _detect_columns_from_header(file_bytes)
    subj_col = find_column(all_cols, SUBJECT_ALIASES)
    lc_col = find_column(all_cols, LC_ALIASES)
    title_col = find_column(all_cols, TITLE_ALIASES)
    weight_col = find_column(all_cols, WEIGHT_ALIASES)

    needed_cols = [c for c in [subj_col, lc_col, title_col, weight_col] if c]
    if not needed_cols:
        needed_cols = None

    with st.spinner(f"Loading {uploaded_file.name}..."):
        df = _load_csv_chunked(file_bytes, uploaded_file.name, cols_to_keep=needed_cols)

    st.success(f"✅ Loaded **{len(df):,}** records from *{uploaded_file.name}*")

    subj_col = find_column(df, SUBJECT_ALIASES)
    lc_col = find_column(df, LC_ALIASES)
    title_col = find_column(df, TITLE_ALIASES)
    weight_col = find_column(df, WEIGHT_ALIASES)

    with st.expander("🔍 Column Detection", expanded=False):
        st.write(f"Subjects: `{subj_col}` · LC: `{lc_col}` · "
                 f"Title: `{title_col}` · Weight: `{weight_col}`")

    if not subj_col and not lc_col:
        st.error("❌ Could not find a Subjects or LC Classification column.")
        return

    if weight_col:
        df['_weight'] = pd.to_numeric(df[weight_col], errors='coerce').fillna(1)
        weight_label = weight_col
    else:
        df['_weight'] = 1.0
        weight_label = "Title Count"

    if lc_col:
        df['_lc_main'], df['_lc_sub'] = _extract_lc_vectorized(df[lc_col])
    else:
        df['_lc_main'] = None
        df['_lc_sub'] = None

    # Sidebar settings
    with st.sidebar:
        st.subheader("⚙️ Analysis Settings")
        weight_options = ["Title count (1 per title)"]
        if weight_col:
            weight_options.append(f"Usage metric ({weight_label})")
        analysis_mode = st.radio("Weight titles by:", weight_options, index=0, key="prof_mode")
        use_weight = weight_col and "Usage" in analysis_mode

        st.markdown("---")
        st.subheader("📊 Visualizations")
        top_n = st.slider("Top N subjects", 10, 100, 30, 5, key="prof_topn")
        show_sunburst = st.checkbox("LC sunburst", True, key="prof_sun")
        show_treemap = st.checkbox("LC treemap", True, key="prof_tree")
        show_bars = st.checkbox("Top subjects bar chart", True, key="prof_bars")
        show_wordcloud = st.checkbox("Subject word cloud", True, key="prof_wc")
        show_heatmap = st.checkbox("LC × subject heatmap", True, key="prof_heat")
        show_gap = st.checkbox("Gap analysis", True, key="prof_gap")
        show_detail = st.checkbox("Title detail table", False, key="prof_detail")

        # Word cloud sub-options (only shown when word cloud is on)
        if show_wordcloud:
            with st.expander("☁️ Word cloud options"):
                wc_max_words = st.slider("Max words", 20, 200, 100, 10, key="prof_wc_max")
                wc_min_len = st.slider("Min word length", 1, 10, 3, key="prof_wc_min")
                wc_color = st.selectbox(
                    "Color scheme",
                    ["viridis", "plasma", "inferno", "magma", "cividis", "twilight", "rainbow"],
                    key="prof_wc_color"
                )
        else:
            wc_max_words, wc_min_len, wc_color = 100, 3, "viridis"

        if lc_col:
            st.markdown("---")
            st.subheader("🔎 Filter by LC Class")
            avail = sorted(df['_lc_main'].dropna().unique())
            labels = [f"{c} – {LC_CLASSES.get(c, '?')}" for c in avail]
            sel_labels = st.multiselect("Include:", labels, default=labels, key="prof_lc_filter")
            sel_classes = [l.split(' –')[0] for l in sel_labels]
        else:
            sel_classes = None

    if st.button("🔍 Analyze Collection", type="primary", use_container_width=True, key="prof_run"):
        w_key = '_weight' if use_weight else None
        pbar = st.progress(0, "Starting analysis...")
        results = _profiler_run_analysis(df, subj_col, lc_col, title_col, w_key, sel_classes, pbar)
        st.session_state['prof_results'] = results
        if sel_classes is not None and lc_col:
            mask = df['_lc_main'].isin(sel_classes) | df['_lc_main'].isna()
            st.session_state['prof_filtered_idx'] = df.index[mask]
        else:
            st.session_state['prof_filtered_idx'] = df.index
        st.session_state['prof_settings'] = {
            'weight_label': weight_label if use_weight else 'Title Count',
            'top_n_subjects': top_n,
            'show_sunburst': show_sunburst, 'show_treemap': show_treemap,
            'show_subject_bars': show_bars,
            'show_wordcloud': show_wordcloud,
            'wc_max_words': wc_max_words, 'wc_min_len': wc_min_len, 'wc_color': wc_color,
            'show_heatmap': show_heatmap,
            'show_gap_analysis': show_gap, 'show_detail_table': show_detail,
        }
        st.success("✅ Analysis complete!")

    if st.session_state['prof_results']:
        _profiler_display_results(
            st.session_state['prof_results'],
            st.session_state.get('prof_settings', {
                'weight_label': 'Title Count', 'top_n_subjects': 30,
                'show_sunburst': True, 'show_treemap': True,
                'show_subject_bars': True, 'show_wordcloud': True,
                'wc_max_words': 100, 'wc_min_len': 3, 'wc_color': 'viridis',
                'show_heatmap': True,
                'show_gap_analysis': True, 'show_detail_table': False,
            }),
            df,
            st.session_state.get('prof_filtered_idx', df.index)
        )


# =====================================================================
# =====================================================================
# TOOL 2: USAGE & SUBSCRIPTION ANALYZER
# =====================================================================
# "What's being used and what isn't?"
# Merged from: app(3).py (COUNTER dashboard) + print circulation mode
# =====================================================================
# =====================================================================

@st.cache_data(show_spinner=False)
def _load_counter_csv(file_bytes, filename):
    """Load a COUNTER 5 TR CSV, stripping the standard 13-row metadata header."""
    # Try common skiprows values since vendors vary slightly
    last_err = None
    for skip in [13, 14, 12, 15]:
        try:
            df = pd.read_csv(BytesIO(file_bytes), skiprows=skip, encoding='utf-8-sig',
                             low_memory=False)
            df.columns = df.columns.str.strip()
            # Sanity check: COUNTER should have Title + Metric_Type or similar
            if 'Title' in df.columns and any(
                c in df.columns for c in ['Metric_Type', 'Reporting Period_Total',
                                           'Reporting_Period_Total']
            ):
                return df, skip
        except Exception as e:
            last_err = e
            continue
    # Fallback: read without skipping
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False)
        df.columns = df.columns.str.strip()
        return df, 0
    except Exception:
        raise last_err or ValueError("Could not parse COUNTER CSV.")


@st.cache_data(show_spinner=False)
def _load_print_csv(file_bytes, filename):
    """Load a print circulation CSV (flat file, no metadata header)."""
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False)
    except Exception:
        df = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _identify_month_columns(df):
    """Identify columns that look like 'Jan-2025' or 'Jan 2025'."""
    return [c for c in df.columns
            if re.match(r'^[A-Za-z]{3}[- ]\d{4}$', c)]


def _render_counter_mode():
    """COUNTER 5 Title Report analysis."""
    st.markdown(
        "Upload a **COUNTER 5 Title Report (TR)** CSV to analyze e-resource usage, "
        "identify cancellation candidates, and track monthly trends."
    )
    uploaded_file = st.file_uploader(
        "Upload your COUNTER 5 TR CSV", type=["csv"], key="usage_counter_upload"
    )
    if uploaded_file is None:
        with st.expander("📖 How to use this mode", expanded=True):
            st.markdown("""
            Upload a standard COUNTER 5 TR export (from EBSCO, ProQuest, Springer, etc.).
            The loader auto-skips the metadata header.

            **Typical workflow:**
            1. Load the file and pick a metric (usually `Unique_Item_Requests`)
            2. Review the **Top Titles** tab to see your workhorses
            3. Check **Cancellation Review** for underperformers
            4. Use **Publisher Summary** to evaluate whole packages
            5. Look at **Monthly Trends** to spot seasonality or decline
            """)
        return

    try:
        file_bytes = uploaded_file.getvalue()
        df_raw, skip_used = _load_counter_csv(file_bytes, uploaded_file.name)
        st.success(f"✅ Loaded **{len(df_raw):,}** rows from *{uploaded_file.name}* "
                   f"(skipped {skip_used} metadata rows)")

        # Detect columns
        month_cols = _identify_month_columns(df_raw)
        total_col = next((c for c in df_raw.columns
                         if c.lower().replace(' ', '_') in
                         ('reporting_period_total',)), None)
        if total_col is None:
            total_col = next((c for c in df_raw.columns
                             if 'total' in c.lower() and 'period' in c.lower()), None)

        with st.expander("🔍 Column Detection", expanded=False):
            st.write(f"**Total column:** `{total_col}`")
            st.write(f"**Month columns detected:** {len(month_cols)} "
                     f"({month_cols[0] if month_cols else 'none'} … "
                     f"{month_cols[-1] if month_cols else 'none'})")
            st.write(f"**All columns:** {list(df_raw.columns)}")

        if total_col is None:
            st.error("❌ Could not find a 'Reporting Period Total' column. "
                     "This may not be a standard COUNTER 5 TR file.")
            return

        # Sidebar filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔎 Counter Filters")

        if 'Metric_Type' in df_raw.columns:
            metric_options = sorted(df_raw['Metric_Type'].dropna().unique())
            default_idx = (list(metric_options).index('Unique_Item_Requests')
                           if 'Unique_Item_Requests' in metric_options else 0)
            selected_metric = st.sidebar.selectbox(
                "Metric Type", metric_options, index=default_idx, key="usage_metric"
            )
            df_raw = df_raw[df_raw['Metric_Type'] == selected_metric].copy()
        else:
            selected_metric = "All (Metric_Type column missing)"

        data_type_col = next((c for c in ['Data Type', 'Data_Type'] if c in df_raw.columns), None)
        if data_type_col:
            data_types = sorted(df_raw[data_type_col].dropna().unique())
            selected_types = st.sidebar.multiselect(
                "Data Type", data_types, default=data_types, key="usage_dtype"
            )
            df_filtered = df_raw[df_raw[data_type_col].isin(selected_types)].copy()
        else:
            df_filtered = df_raw.copy()

        if 'Publisher' in df_filtered.columns:
            publishers = sorted(df_filtered['Publisher'].dropna().unique())
            selected_pubs = st.sidebar.multiselect(
                "Publisher", publishers, default=publishers, key="usage_pub"
            )
            df_filtered = df_filtered[df_filtered['Publisher'].isin(selected_pubs)]

        # KPIs
        st.markdown("---")
        st.subheader(f"📊 Overview — {selected_metric}")
        total_usage = pd.to_numeric(df_filtered[total_col], errors='coerce').fillna(0).sum()
        unique_titles = df_filtered['Title'].nunique() if 'Title' in df_filtered.columns else 0
        avg_usage = total_usage / unique_titles if unique_titles > 0 else 0
        zero_use = (pd.to_numeric(df_filtered[total_col], errors='coerce').fillna(0) == 0).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Usage", f"{int(total_usage):,}")
        k2.metric("Unique Titles", f"{unique_titles:,}")
        k3.metric("Avg Usage / Title", f"{avg_usage:.1f}")
        k4.metric("Zero-Use Titles", f"{zero_use:,}")

        # Ensure numeric for analysis
        df_filtered = df_filtered.copy()
        df_filtered['_total'] = pd.to_numeric(df_filtered[total_col], errors='coerce').fillna(0)

        # Analysis tabs
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Top Titles", "✂️ Cancellation Review",
            "🏢 Publisher Summary", "📈 Monthly Trends"
        ])

        with tab1:
            top_n = st.slider("Show Top N Titles", 5, 100, 25, key="usage_topn")
            top_cols = ['Title', '_total']
            if 'Publisher' in df_filtered.columns:
                top_cols.insert(1, 'Publisher')
            top_titles = df_filtered.nlargest(top_n, '_total')[top_cols].rename(
                columns={'_total': selected_metric}
            )
            fig_top = px.bar(
                top_titles, x=selected_metric, y='Title', orientation='h',
                title=f"Top {top_n} Titles by {selected_metric}",
                color=selected_metric,
                color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                hover_data={'Publisher': True} if 'Publisher' in top_titles.columns else None
            )
            fig_top.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(450, top_n * 22)
            )
            st.plotly_chart(fig_top, use_container_width=True)
            st.download_button("📥 Top Titles (CSV)",
                               top_titles.to_csv(index=False),
                               "top_titles.csv", "text/csv", key="usage_dl_top")

        with tab2:
            st.info("📌 Review titles with low usage for potential cancellation or renegotiation.")
            threshold = st.number_input(
                "Low-Usage Threshold (total for reporting period)",
                min_value=0, value=5, key="usage_threshold"
            )
            low_cols = ['Title', '_total']
            if 'Publisher' in df_filtered.columns:
                low_cols.insert(1, 'Publisher')
            if data_type_col:
                low_cols.append(data_type_col)
            low_use_df = df_filtered[df_filtered['_total'] <= threshold][low_cols].sort_values('_total')
            low_use_df = low_use_df.rename(columns={'_total': selected_metric})

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Titles ≤ Threshold", f"{len(low_use_df):,}")
            cc2.metric("% of Filtered", f"{len(low_use_df)/max(1,len(df_filtered))*100:.1f}%")
            cc3.metric("Usage Lost if Cancelled",
                       f"{int(low_use_df[selected_metric].sum()):,}")

            st.dataframe(low_use_df, use_container_width=True, height=400)
            st.download_button("📥 Cancellation Review List (CSV)",
                               low_use_df.to_csv(index=False),
                               "cancellation_review.csv", "text/csv",
                               key="usage_dl_cancel")

        with tab3:
            if 'Publisher' in df_filtered.columns:
                pub_summary = df_filtered.groupby('Publisher').agg(
                    **{
                        'Title Count': ('Title', 'nunique'),
                        'Total Usage': ('_total', 'sum'),
                    }
                ).reset_index()
                pub_summary['Usage Density'] = (
                    pub_summary['Total Usage'] / pub_summary['Title Count']
                ).round(1)
                pub_summary['Zero-Use Titles'] = df_filtered.assign(
                    _zero=(df_filtered['_total'] == 0).astype(int)
                ).groupby('Publisher')['_zero'].sum().reindex(pub_summary['Publisher']).values
                pub_summary = pub_summary.sort_values('Total Usage', ascending=False)

                st.dataframe(pub_summary, use_container_width=True, hide_index=True, height=500)
                st.download_button("📥 Publisher Summary (CSV)",
                                   pub_summary.to_csv(index=False),
                                   "publisher_summary.csv", "text/csv",
                                   key="usage_dl_pub")
            else:
                st.info("No Publisher column in this file.")

        with tab4:
            if month_cols:
                # Melt only month columns that are actually in the filtered df
                id_vars = ['Title']
                if 'Publisher' in df_filtered.columns:
                    id_vars.append('Publisher')
                present_months = [c for c in month_cols if c in df_filtered.columns]
                if not present_months:
                    st.info("No month columns found in filtered data.")
                else:
                    df_melted = df_filtered.melt(
                        id_vars=id_vars, value_vars=present_months,
                        var_name='Month', value_name='Usage'
                    )
                    df_melted['Usage'] = pd.to_numeric(df_melted['Usage'], errors='coerce').fillna(0)
                    # Support both Jan-2025 and Jan 2025
                    df_melted['Month'] = pd.to_datetime(
                        df_melted['Month'].str.replace(' ', '-'), format='%b-%Y', errors='coerce'
                    )
                    monthly_trend = df_melted.groupby('Month', as_index=False)['Usage'].sum()
                    fig_trend = px.line(
                        monthly_trend, x='Month', y='Usage', markers=True,
                        title=f"Monthly {selected_metric} — All Selected Titles",
                    )
                    fig_trend.update_traces(line_color='#285C4D')
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # Also offer top-title breakdown
                    st.markdown("**Monthly trend for top 5 titles:**")
                    top5_titles = df_filtered.nlargest(5, '_total')['Title'].tolist()
                    tdf = df_melted[df_melted['Title'].isin(top5_titles)]
                    if not tdf.empty:
                        fig_tt = px.line(
                            tdf.groupby(['Month', 'Title'], as_index=False)['Usage'].sum(),
                            x='Month', y='Usage', color='Title', markers=True,
                            title="Monthly Usage — Top 5 Titles"
                        )
                        st.plotly_chart(fig_tt, use_container_width=True)
            else:
                st.info("No month columns (e.g., `Jan-2025`) detected in this file.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("Ensure you are uploading a standard COUNTER 5 TR CSV. "
                "If the file has an unusual structure, try exporting a fresh copy from the vendor.")


def _render_print_mode():
    """Print circulation analysis."""
    st.markdown(
        "Upload a **print circulation export** to identify high- and low-circulating "
        "titles for weeding review, collection rotation, or replacement decisions."
    )
    uploaded_file = st.file_uploader(
        "Upload your print circulation CSV", type=["csv"], key="usage_print_upload"
    )
    if uploaded_file is None:
        with st.expander("📖 How to use this mode", expanded=True):
            st.markdown("""
            Upload a flat CSV with columns like:
            - `Title` *(required)*
            - `Loans` / `Checkouts` / `Circulation` *(one required as the usage metric)*
            - `Author`, `LC Classification`, `Location`, `Publication Year` *(optional, enable more analysis)*

            **Typical workflow:**
            1. Load the file; the loader auto-detects your usage column
            2. Review **Top Titles** to see your workhorses
            3. Check **Weeding Review** for low/no-circulation candidates
            4. Use **LC Breakdown** to see which areas of the collection are pulling weight
            """)
        return

    try:
        file_bytes = uploaded_file.getvalue()
        df = _load_print_csv(file_bytes, uploaded_file.name)
        st.success(f"✅ Loaded **{len(df):,}** records from *{uploaded_file.name}*")

        # Detect columns
        title_col = find_column(df, TITLE_ALIASES)
        weight_col = find_column(df, WEIGHT_ALIASES)
        lc_col = find_column(df, LC_ALIASES)
        author_col = find_column(df, ['Author', 'author', 'AUTHOR', 'Creator'])
        location_col = find_column(df, ['Location', 'Location Name', 'location'])

        with st.expander("🔍 Column Detection", expanded=False):
            st.write(f"Title: `{title_col}` · Usage: `{weight_col}` · "
                     f"LC: `{lc_col}` · Author: `{author_col}` · Location: `{location_col}`")

        if not title_col:
            st.error("❌ Need a Title column.")
            return
        if not weight_col:
            st.error("❌ Need a usage column (Loans, Checkouts, Circulation, etc.).")
            return

        df['_usage'] = pd.to_numeric(df[weight_col], errors='coerce').fillna(0)

        # Sidebar filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔎 Print Filters")
        if location_col:
            locs = sorted(df[location_col].dropna().unique())
            selected_locs = st.sidebar.multiselect(
                "Location", locs, default=locs, key="usage_p_loc"
            )
            df = df[df[location_col].isin(selected_locs)].copy()

        if lc_col:
            df['_lc_main'] = df[lc_col].apply(extract_lc_prefix)
            lc_avail = sorted(df['_lc_main'].dropna().unique())
            lc_labels = [f"{c} – {LC_CLASSES.get(c, '?')}" for c in lc_avail]
            sel_lc_labels = st.sidebar.multiselect(
                "LC Class", lc_labels, default=lc_labels, key="usage_p_lc"
            )
            sel_lc = [l.split(' –')[0] for l in sel_lc_labels]
            df = df[df['_lc_main'].isin(sel_lc) | df['_lc_main'].isna()].copy()

        # KPIs
        st.markdown("---")
        st.subheader(f"📊 Overview — {weight_col}")
        total_loans = df['_usage'].sum()
        zero_use = (df['_usage'] == 0).sum()
        avg_loans = total_loans / max(1, len(df))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Records", f"{len(df):,}")
        k2.metric(f"Total {weight_col}", f"{int(total_loans):,}")
        k3.metric(f"Avg {weight_col} / Title", f"{avg_loans:.1f}")
        k4.metric("Zero-Use Titles", f"{zero_use:,}")

        # Analysis tabs
        st.markdown("---")
        tabs = ["🏆 Top Titles", "✂️ Weeding Review"]
        if lc_col:
            tabs.append("📚 LC Breakdown")
        if author_col:
            tabs.append("✍️ Author Summary")
        tab_objs = st.tabs(tabs)

        with tab_objs[0]:
            top_n = st.slider("Show Top N Titles", 5, 100, 25, key="usage_p_topn")
            display_cols = [title_col, '_usage']
            if author_col:
                display_cols.insert(1, author_col)
            if lc_col:
                display_cols.append(lc_col)
            top_titles = df.nlargest(top_n, '_usage')[display_cols].rename(
                columns={'_usage': weight_col}
            )
            fig_top = px.bar(
                top_titles, x=weight_col, y=title_col, orientation='h',
                title=f"Top {top_n} Titles by {weight_col}",
                color=weight_col,
                color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']]
            )
            fig_top.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(450, top_n * 22)
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with tab_objs[1]:
            st.info("📌 Review titles with low or zero circulation for potential weeding, "
                    "off-site storage, or replacement.")
            threshold = st.number_input(
                f"Low-Usage Threshold ({weight_col})", min_value=0, value=0, key="usage_p_thr"
            )
            low_cols = [title_col, '_usage']
            if author_col:
                low_cols.insert(1, author_col)
            if lc_col:
                low_cols.append(lc_col)
            if location_col:
                low_cols.append(location_col)
            low_use_df = df[df['_usage'] <= threshold][low_cols].sort_values('_usage')
            low_use_df = low_use_df.rename(columns={'_usage': weight_col})

            cc1, cc2 = st.columns(2)
            cc1.metric("Titles ≤ Threshold", f"{len(low_use_df):,}")
            cc2.metric("% of Collection", f"{len(low_use_df)/max(1,len(df))*100:.1f}%")

            st.dataframe(low_use_df, use_container_width=True, height=400)
            st.download_button("📥 Weeding Review List (CSV)",
                               low_use_df.to_csv(index=False),
                               "weeding_review.csv", "text/csv",
                               key="usage_p_dl_weed")

        if lc_col:
            with tab_objs[2]:
                lc_summary = df.groupby('_lc_main').agg(
                    **{
                        'Title Count': (title_col, 'count'),
                        f'Total {weight_col}': ('_usage', 'sum'),
                    }
                ).reset_index().rename(columns={'_lc_main': 'LC Class'})
                lc_summary[f'Avg {weight_col} / Title'] = (
                    lc_summary[f'Total {weight_col}'] / lc_summary['Title Count']
                ).round(2)
                lc_summary['Description'] = lc_summary['LC Class'].map(
                    lambda c: LC_CLASSES.get(c, '?')
                )
                lc_summary = lc_summary[['LC Class', 'Description', 'Title Count',
                                         f'Total {weight_col}', f'Avg {weight_col} / Title']]
                lc_summary = lc_summary.sort_values(f'Total {weight_col}', ascending=False)
                st.dataframe(lc_summary, use_container_width=True,
                             hide_index=True, height=500)
                st.download_button("📥 LC Breakdown (CSV)",
                                   lc_summary.to_csv(index=False),
                                   "lc_circulation_breakdown.csv", "text/csv",
                                   key="usage_p_dl_lc")

        if author_col:
            idx = 3 if lc_col else 2
            with tab_objs[idx]:
                auth_summary = df.groupby(author_col).agg(
                    **{
                        'Title Count': (title_col, 'count'),
                        f'Total {weight_col}': ('_usage', 'sum'),
                    }
                ).reset_index().sort_values(f'Total {weight_col}', ascending=False).head(100)
                st.markdown("**Top 100 authors by total circulation:**")
                st.dataframe(auth_summary, use_container_width=True,
                             hide_index=True, height=500)
                st.download_button("📥 Author Summary (CSV)",
                                   auth_summary.to_csv(index=False),
                                   "author_circulation.csv", "text/csv",
                                   key="usage_p_dl_auth")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")


def page_usage_analyzer():
    """Tool 2: Usage & Subscription Analyzer."""
    st.header("📈 Usage & Subscription Analyzer")
    st.markdown(
        "**What's being used and what isn't?** Title-level usage analysis for "
        "renewal, cancellation, and weeding decisions."
    )
    _decision_box(
        "When to use this tool",
        "- **Collections:** Annual database renewals, print weeding by low circulation, "
        "identifying single-title-driven subscriptions (candidates for swap to pay-per-view "
        "or title-level purchase), publisher package evaluation.\n"
        "- **Instruction:** Rarely — this is primarily a purchasing/weeding tool, not "
        "an instructional one.\n"
        "- **Outreach:** Evidence for faculty conversations (\"this database gets 12 uses "
        "a year across your whole department — can we talk alternatives?\"), value reports "
        "to administration, package-vs-title comparisons for liaison meetings."
    )
    st.markdown("---")

    mode = st.radio(
        "Select data source:",
        ["📊 COUNTER 5 (e-resources)", "📚 Print Circulation"],
        horizontal=True, key="usage_mode"
    )
    st.markdown("---")

    if mode == "📊 COUNTER 5 (e-resources)":
        _render_counter_mode()
    else:
        _render_print_mode()


# =====================================================================
# =====================================================================
# TOOL 3: ACQUISITION RECOMMENDATION SCORER
# =====================================================================
# "What should we buy next?"
# Score candidate books against checkout history + optional faculty interests
# =====================================================================
# =====================================================================

def _ensure_nltk():
    """Download NLTK data if needed."""
    if not NLTK_AVAILABLE:
        return False
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return True


def normalize_author(author):
    """Normalize author for lookup, generating reversed forms."""
    if pd.isna(author) or not isinstance(author, str):
        return set()
    norm = normalize_text(author)
    if len(norm) <= 2:
        return set()
    candidates = {norm}
    if "," in author:
        parts = [normalize_text(p) for p in author.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            candidates.add(f"{parts[1]} {parts[0]}".strip())
    return candidates


REQUIRED_CHECKOUT_COLS = {"title", "checkouts"}
RECOMMENDED_CHECKOUT_COLS = {"author", "subjects", "lc_classification"}
REQUIRED_REC_COLS = {"title"}
RECOMMENDED_REC_COLS = {"author", "subjects", "lc_classification"}


def _suggest(col, candidates, threshold=0.75):
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, col.lower(), c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None


def validate_columns(df, required, recommended, file_label):
    actual = set(df.columns.str.lower())
    df.columns = df.columns.str.lower()
    warnings = []
    valid = True
    for col in required:
        if col not in actual:
            suggestion = _suggest(col, actual)
            hint = f" Did you mean **`{suggestion}`**?" if suggestion else ""
            st.error(f"❌ **{file_label}** is missing required column `{col}`.{hint}")
            valid = False
    for col in recommended:
        if col not in actual:
            suggestion = _suggest(col, actual)
            hint = f" (closest match: `{suggestion}`)" if suggestion else ""
            warnings.append(f"`{col}` not found{hint} — scores for this factor will be 0")
    return valid, warnings


def validate_checkouts_numeric(df):
    if "checkouts" not in df.columns:
        return df
    original = df["checkouts"].copy()
    df["checkouts"] = pd.to_numeric(df["checkouts"], errors="coerce").fillna(0)
    bad = original[df["checkouts"] == 0][original != 0].count()
    if bad > 0:
        st.warning(f"⚠️ {bad} rows in the checkouts column had non-numeric values and were set to 0.")
    return df


def consolidate_checkouts(df):
    key_cols = [c for c in ["title", "author"] if c in df.columns]
    if not key_cols:
        return df
    rows_before = len(df)
    multi_year_titles = df[df.duplicated(subset=key_cols, keep=False)][key_cols[0]].nunique()
    if multi_year_titles == 0:
        return df
    agg_rules = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if col == "checkouts" or pd.api.types.is_numeric_dtype(df[col]):
            agg_rules[col] = "sum"
        else:
            agg_rules[col] = "first"
    consolidated = df.groupby(key_cols, as_index=False, sort=False).agg(agg_rules)
    rows_after = len(consolidated)
    st.info(f"📅 **Multi-year data:** {multi_year_titles} title(s) consolidated "
            f"({rows_before} rows → {rows_after} unique titles). Checkouts summed.")
    return consolidated.reset_index(drop=True)


def check_duplicates_recommendations(df):
    key_cols = [c for c in ["title", "author"] if c in df.columns]
    if not key_cols:
        return df
    dupes = df.duplicated(subset=key_cols, keep="first").sum()
    if dupes > 0:
        st.warning(f"⚠️ Recommendations: {dupes} duplicate row(s) removed.")
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return df


def extract_all_subjects(df):
    subject_counts = defaultdict(int)
    if "subjects" not in df.columns:
        return subject_counts
    for subjects_str in df["subjects"].dropna():
        for subject in split_subjects(subjects_str):
            if subject:
                subject_counts[subject] += 1
    return dict(subject_counts)


# Synonym map (kept in Tool 3 since only the scorer uses it)
BUILTIN_SYNONYM_GROUPS = {
    "radical_extreme": ["radical", "extreme", "extremist", "fringe", "militant", "revolutionary"],
    "conservative": ["conservative", "traditional", "right-wing", "reactionary"],
    "progressive_liberal": ["progressive", "liberal", "left-wing", "reformist"],
    "equity_justice": ["equity", "equality", "justice", "fairness", "parity"],
    "inclusion_diversity": ["inclusion", "diversity", "belonging", "representation", "multiculturalism"],
    "discrimination": ["discrimination", "bias", "prejudice", "racism", "sexism", "oppression"],
    "climate_change": ["climate change", "global warming", "greenhouse", "carbon", "emissions"],
    "environment_ecology": ["environment", "ecology", "ecosystem", "biodiversity", "conservation", "sustainability"],
    "mental_health": ["mental health", "mental illness", "psychiatric", "psychological", "wellbeing"],
    "chronic_disease": ["chronic disease", "chronic illness", "long-term condition", "comorbidity"],
    "infectious_disease": ["infectious disease", "epidemic", "pandemic", "outbreak", "pathogen"],
    "violence": ["violence", "aggression", "assault", "brutality", "coercion"],
    "war_conflict": ["war", "conflict", "warfare", "combat", "armed conflict", "insurgency"],
    "migration": ["migration", "immigration", "emigration", "diaspora", "mobility"],
    "refugee": ["refugee", "asylum seeker", "displaced person", "exile"],
    "artificial_intelligence": ["artificial intelligence", "machine learning", "deep learning", "AI", "neural network"],
    "data_privacy": ["privacy", "data protection", "surveillance", "tracking"],
    "poverty_inequality": ["poverty", "inequality", "deprivation", "disadvantage", "underserved"],
    "pedagogy_teaching": ["pedagogy", "teaching", "instruction", "education", "curriculum", "learning"],
    "gender_identity": ["gender", "gender identity", "transgender", "nonbinary"],
    "sexuality": ["sexuality", "sexual orientation", "LGBTQ", "queer"],
    "race_ethnicity": ["race", "ethnicity", "racial", "ethnic"],
    "religion_faith": ["religion", "faith", "spirituality", "theology"],
}


def build_synonym_map(stemmer, user_overrides_df=None):
    groups = {label: list(terms) for label, terms in BUILTIN_SYNONYM_GROUPS.items()}
    if user_overrides_df is not None and not user_overrides_df.empty:
        for _, row in user_overrides_df.iterrows():
            term = str(row.get("term", "")).strip()
            label = str(row.get("group_label", "")).strip()
            if term and label:
                groups.setdefault(label, []).append(term)
    synonym_map = {}
    for label, terms in groups.items():
        for term in terms:
            norm = normalize_text(term)
            for word in norm.split():
                stemmed = stemmer.stem(word)
                if len(stemmed) > 2:
                    synonym_map.setdefault(stemmed, label)
    return synonym_map


def apply_synonym_map(terms, synonym_map):
    return [synonym_map.get(t, t) for t in terms]


class FacultyScorer:
    def __init__(self, faculty_df, stemmer, synonym_map):
        self.stemmer = stemmer
        self.synonym_map = synonym_map
        self.faculty_index = self._build_index(faculty_df)

    def _tokenize(self, text):
        norm = normalize_text(text) if text else ""
        if not norm:
            return []
        stemmed = [self.stemmer.stem(w) for w in norm.split() if len(w) > 2]
        return apply_synonym_map(stemmed, self.synonym_map)

    def _build_index(self, faculty_df):
        index = []
        for _, row in faculty_df.iterrows():
            name = str(row.get("name", "")).strip()
            dept = str(row.get("department", "")).strip()
            interests_raw = str(row.get("research_interests", ""))
            tokens = set(self._tokenize(interests_raw))
            if tokens:
                index.append({"name": name, "department": dept, "tokens": tokens})
        return index

    def score(self, recommendation):
        raw_subjects = recommendation.get("subjects", "")
        raw_title = recommendation.get("title", "")
        combined = f"{raw_subjects} {raw_title}"
        rec_tokens = set(self._tokenize(combined))
        if not rec_tokens or not self.faculty_index:
            return 0.0, ""
        best_score = 0.0
        best_label = ""
        for faculty in self.faculty_index:
            fac_tokens = faculty["tokens"]
            if not fac_tokens:
                continue
            intersection = rec_tokens & fac_tokens
            union = rec_tokens | fac_tokens
            if not union:
                continue
            jaccard = len(intersection) / len(union)
            scaled = min(jaccard * 300, 100.0)
            if scaled > best_score:
                best_score = scaled
                dept_str = f" ({faculty['department']})" if faculty["department"] else ""
                best_label = f"{faculty['name']}{dept_str}"
        return round(best_score, 2), best_label


class RecommendationScorer:
    def __init__(self, checkouts_df, synonym_map=None):
        self.checkouts_df = checkouts_df
        self.stemmer = SnowballStemmer("english")
        self.synonym_map = synonym_map or {}
        self.total_docs = len(checkouts_df)
        self.semantic_groups = self._build_semantic_groups()
        self.author_checkout_map = self._build_author_map()
        self.lc_checkout_map = self._build_lc_map()
        self.subject_terms = self._extract_subject_terms_enhanced()
        self.term_frequencies = self._calculate_term_frequencies()

    def _build_semantic_groups(self):
        groups = {
            "computer_science": ["comput", "programm", "softwar", "algorithm", "code"],
            "artificial_intelligence": ["artifici", "intellig", "machin", "learn", "neural", "deep", "ai"],
            "data_analytics": ["data", "analysi", "analyt", "statist", "visual", "databas"],
            "psychology": ["psycholog", "mental", "health", "behavior", "cognit", "psychiatr"],
            "sociology": ["sociolog", "social", "cultur", "commun", "society"],
            "economics": ["econom", "market", "trade", "finance", "financi", "busi"],
            "political_science": ["politic", "govern", "polici", "democrat", "elect", "legisl"],
            "history": ["histor", "histori", "past", "ancient", "mediev", "modern", "war"],
            "philosophy": ["philosoph", "ethic", "moral", "metaphys", "epistemolog"],
            "literature": ["literatur", "novel", "fiction", "poetri", "drama", "narrat"],
            "education": ["educ", "teach", "learn", "pedagog", "curriculum", "school"],
            "law": ["law", "legal", "court", "justic", "judg", "attorney"],
            "medicine": ["medicin", "medic", "health", "clinic", "hospit", "treatment", "diseas"],
            "environmental": ["environ", "climat", "ecolog", "sustain", "conserv", "ecosyst"],
            "biology": ["biolog", "life", "scienc", "organ", "cell", "geneti"],
            "library_science": ["librari", "inform", "catalog", "bibliograph", "archiv", "collect"],
            "gender_studies": ["gender", "feminis", "women", "masculin", "queer", "lgbt"],
            "diversity": ["divers", "inclus", "equiti", "racial", "ethnic", "multicultural"],
        }
        term_to_group = {}
        for group_id, terms in groups.items():
            for term in terms:
                term_to_group.setdefault(term, []).append(group_id)
        return {"groups": groups, "term_to_group": term_to_group}

    def _build_author_map(self):
        author_map = defaultdict(list)
        for _, row in self.checkouts_df.iterrows():
            for candidate in normalize_author(row.get("author", "")):
                author_map[candidate].append(row.get("checkouts", 0))
        return dict(author_map)

    def _build_lc_map(self):
        lc_map = defaultdict(list)
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get("lc_classification")):
                lc_prefix = extract_lc_prefix(row["lc_classification"])
                if lc_prefix:
                    lc_map[lc_prefix].append(row.get("checkouts", 0))
        return dict(lc_map)

    def _extract_subject_terms_enhanced(self):
        all_terms = []
        doc_term_counts = defaultdict(set)
        unique_subject_docs = set()
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get("subjects")):
                subjects = split_subjects(str(row["subjects"]))
                checkouts = row.get("checkouts", 0)
                for i, subject in enumerate(subjects):
                    unique_subject_docs.add(subject)
                    hw = 1.0 if i == 0 else 0.7
                    for term in self._tokenize_and_stem(subject):
                        all_terms.append((term, checkouts * hw, False))
                        doc_term_counts[term].add(subject)
                    for bigram in self._extract_bigrams(subject):
                        all_terms.append((bigram, checkouts * hw * 1.3, True))
                        doc_term_counts[bigram].add(subject)
        total_subject_docs = max(len(unique_subject_docs), 1)
        term_checkouts = defaultdict(list)
        for term, checkout_count, _ in all_terms:
            term_checkouts[term].append(checkout_count)
        term_scores = {}
        for term, counts in term_checkouts.items():
            avg = sum(counts) / len(counts)
            docs_with = len(doc_term_counts[term])
            idf = np.log(total_subject_docs / (1 + docs_with))
            term_scores[term] = avg * (1 + idf * 0.3)
        return term_scores

    def _extract_bigrams(self, text):
        if not text:
            return []
        words = [w for w in text.split() if len(w) > 2]
        return [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]

    def _calculate_term_frequencies(self):
        tf = Counter()
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get("subjects")):
                norm = normalize_text(str(row["subjects"]))
                tf.update(self._tokenize_and_stem(norm))
        return tf

    def _tokenize_and_stem(self, text):
        norm = normalize_text(text) if text else ""
        if not norm:
            return []
        stemmed = [self.stemmer.stem(w) for w in norm.split() if len(w) > 2]
        return apply_synonym_map(stemmed, self.synonym_map)

    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(self.stemmer.stem(lemma.name().lower()))
        return synonyms

    def _get_semantic_matches(self, term):
        matches = []
        ttg = self.semantic_groups["term_to_group"]
        if term in ttg:
            groups = ttg[term]
            for gid in groups:
                for gt in self.semantic_groups["groups"][gid]:
                    if gt in self.subject_terms:
                        if gt in ttg:
                            shared = set(groups) & set(ttg[gt])
                            strength = len(shared) * 0.85
                        else:
                            strength = 0.85
                        matches.append((gt, self.subject_terms[gt], strength))
        return matches

    def _fuzzy_match_terms(self, term, threshold=0.80):
        max_score = 0
        for existing_term in self.subject_terms:
            sim = SequenceMatcher(None, term, existing_term).ratio()
            if sim >= threshold:
                max_score = max(max_score, self.subject_terms[existing_term])
        return max_score

    def _calculate_subject_similarity(self, recommendation):
        raw = recommendation.get("subjects")
        if pd.isna(raw) or not self.subject_terms:
            return 0.0
        norm = normalize_text(str(raw))
        rec_terms = self._tokenize_and_stem(norm)
        rec_bigrams = self._extract_bigrams(norm)
        all_rec = rec_terms + rec_bigrams
        if not all_rec:
            return 0.0
        total_score = 0
        matched = 0
        exact = 0
        for rt in all_rec:
            rec_syns = self._get_synonyms(rt.replace("_", " "))
            rec_syns.add(rt)
            max_ts = 0
            if rt in self.subject_terms:
                max_ts = self.subject_terms[rt] * 1.5
                exact += 1
            if max_ts == 0:
                for syn in rec_syns:
                    if syn in self.subject_terms:
                        max_ts = max(max_ts, self.subject_terms[syn])
            if max_ts == 0:
                for _, ts, strength in self._get_semantic_matches(rt):
                    max_ts = max(max_ts, ts * strength)
            if max_ts == 0:
                max_ts = self._fuzzy_match_terms(rt)
            if max_ts > 0:
                matched += 1
                total_score += max_ts
        if matched == 0:
            return 0.0
        avg = total_score / matched
        max_c = max(self.subject_terms.values())
        coverage = matched / len(all_rec)
        exact_ratio = exact / len(all_rec)
        cw = min(0.6 + 0.4 * coverage + 0.2 * exact_ratio, 1.0)
        return (avg / max_c) * 100 * cw

    def _calculate_lc_score(self, recommendation):
        if pd.isna(recommendation.get("lc_classification")) or not self.lc_checkout_map:
            return 0.0
        lc_prefix = extract_lc_prefix(recommendation["lc_classification"])
        if not lc_prefix or lc_prefix not in self.lc_checkout_map:
            return 0.0
        vals = self.lc_checkout_map[lc_prefix]
        avg = sum(vals) / len(vals)
        max_avg = max(sum(v) / len(v) for v in self.lc_checkout_map.values())
        return (avg / max_avg) * 100

    def _calculate_author_score(self, recommendation):
        candidates = normalize_author(recommendation.get("author", ""))
        if not candidates or not self.author_checkout_map:
            return 0.0
        max_avg = max(sum(v) / len(v) for v in self.author_checkout_map.values())
        best = 0.0
        for c in candidates:
            if c in self.author_checkout_map:
                vals = self.author_checkout_map[c]
                avg = sum(vals) / len(vals)
                best = max(best, (avg / max_avg) * 100)
        return best

    def score_recommendations(self, recommendations_df,
                              subject_weight=0.5, lc_weight=0.3,
                              author_weight=0.2, faculty_weight=0.0,
                              faculty_scorer=None):
        results = []
        for _, rec in recommendations_df.iterrows():
            ss = self._calculate_subject_similarity(rec)
            ls = self._calculate_lc_score(rec)
            aus = self._calculate_author_score(rec)
            fs, mf = 0.0, ""
            if faculty_scorer and faculty_weight > 0:
                fs, mf = faculty_scorer.score(rec)
            likelihood = ss * subject_weight + ls * lc_weight + aus * author_weight + fs * faculty_weight
            d = rec.to_dict()
            d["likelihood_score"] = round(likelihood, 2)
            d["similarity_score"] = round(ss, 2)
            d["checkout_volume_score"] = round(ls, 2)
            d["author_popularity_score"] = round(aus, 2)
            d["faculty_interest_score"] = round(fs, 2)
            d["matched_faculty"] = mf
            results.append(d)
        rdf = pd.DataFrame(results)
        rdf = rdf.sort_values("likelihood_score", ascending=False).reset_index(drop=True)
        return rdf


def generate_report(results_df):
    lines = ["=" * 80, "LIBRARY BOOK RECOMMENDATION REPORT", "=" * 80, "",
             "SUMMARY", "-" * 80,
             f"Total Recommendations Analyzed: {len(results_df)}", ""]
    high = len(results_df[results_df["likelihood_score"] >= 70])
    medium = len(results_df[(results_df["likelihood_score"] >= 40) & (results_df["likelihood_score"] < 70)])
    low = len(results_df[results_df["likelihood_score"] < 40])
    lines += [
        f"High Priority (70-100):   {high} books  ({high/max(1,len(results_df))*100:.1f}%)",
        f"Medium Priority (40-69):  {medium} books  ({medium/max(1,len(results_df))*100:.1f}%)",
        f"Low Priority (0-39):      {low} books  ({low/max(1,len(results_df))*100:.1f}%)",
        "", "TOP 20 RECOMMENDATIONS", "=" * 80, "",
    ]
    for idx, row in results_df.head(20).iterrows():
        lines += [
            f"#{idx + 1}: {row['title']}",
            f"   Author: {row.get('author', 'N/A')}",
            f"   Overall Score: {row['likelihood_score']:.1f}/100",
            f"   - Subject Similarity:    {row['similarity_score']:.1f}",
            f"   - Checkout Volume:       {row['checkout_volume_score']:.1f}",
            f"   - Author Popularity:     {row['author_popularity_score']:.1f}",
            f"   - Faculty Interest:      {row.get('faculty_interest_score', 0.0):.1f}",
            f"   - Matched Faculty:       {row.get('matched_faculty', 'N/A')}", "",
        ]
    return "\n".join(lines)


def page_recommendation_scorer():
    """Tool 3: Acquisition Recommendation Scorer."""
    if not NLTK_AVAILABLE:
        st.error("The `nltk` package is required for this tool. Install with: `pip install nltk`")
        return
    _ensure_nltk()

    st.header("📊 Acquisition Recommendation Scorer")
    st.markdown(
        "**What should we buy next?** Score candidate books against your checkout "
        "history to prioritize purchases."
    )
    _decision_box(
        "When to use this tool",
        "- **Collections:** Evaluating vendor slip lists and GOBI picks, approval-plan "
        "exceptions, triaging faculty requests, flipping DDA candidates to purchase, "
        "reviewing author/publisher lists for standing orders.\n"
        "- **Instruction:** Occasionally — clusters of high-scoring recommendations in "
        "one area can reveal curricular momentum worth a targeted info-lit session.\n"
        "- **Outreach:** Showing faculty *why* a recommendation scored high (with "
        "the faculty-interest score naming their research match) makes this a strong "
        "conversation-starter at liaison meetings."
    )
    st.info("💡 For a broader view of your collection's LC distribution and subject "
            "coverage, use the **Collection Profiler** (first tool in the sidebar).")
    st.markdown("---")

    # File uploads
    st.subheader("📁 Step 1: Upload Your Data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Checkouts File** — Required: `title`, `checkouts`; "
                    "Recommended: `author`, `subjects`, `lc_classification`")
        checkouts_file = st.file_uploader("Upload checkouts CSV", type=["csv"], key="rec_checkouts")
    with c2:
        st.markdown("**Recommendations File** — Required: `title`; "
                    "Recommended: `author`, `subjects`, `lc_classification`")
        recommendations_file = st.file_uploader("Upload recommendations CSV", type=["csv"], key="rec_recs")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Faculty Research Interests** *(optional)* — "
                    "Columns: `name`, `department`, `research_interests`")
        faculty_file = st.file_uploader("Upload faculty CSV", type=["csv"], key="rec_faculty")
    with c4:
        st.markdown("**Custom Synonym Groups** *(optional)* — "
                    "Columns: `term`, `group_label`")
        synonym_file = st.file_uploader("Upload synonym CSV", type=["csv"], key="rec_synonyms")

    if checkouts_file and recommendations_file:
        try:
            with st.spinner("Loading and validating data..."):
                checkouts_df = pd.read_csv(checkouts_file)
                recommendations_df = pd.read_csv(recommendations_file)

            co_valid, co_warns = validate_columns(checkouts_df, REQUIRED_CHECKOUT_COLS,
                                                   RECOMMENDED_CHECKOUT_COLS, "Checkouts file")
            re_valid, re_warns = validate_columns(recommendations_df, REQUIRED_REC_COLS,
                                                   RECOMMENDED_REC_COLS, "Recommendations file")

            if co_warns or re_warns:
                with st.expander("⚠️ Column warnings"):
                    for w in co_warns + re_warns:
                        st.markdown(f"- {w}")
            if not (co_valid and re_valid):
                st.stop()

            checkouts_df = validate_checkouts_numeric(checkouts_df)
            checkouts_df = consolidate_checkouts(checkouts_df)
            recommendations_df = check_duplicates_recommendations(recommendations_df)

            # Faculty
            faculty_df = None
            if faculty_file:
                faculty_df = pd.read_csv(faculty_file)
                faculty_df.columns = faculty_df.columns.str.lower()
                missing_fac = [c for c in ["name", "department", "research_interests"]
                               if c not in faculty_df.columns]
                if missing_fac:
                    st.warning(f"⚠️ Faculty file missing: {missing_fac}. Faculty scoring disabled.")
                    faculty_df = None
                else:
                    st.success(f"✅ Loaded {len(faculty_df)} faculty records")

            # Synonyms
            synonym_overrides_df = None
            if synonym_file:
                synonym_overrides_df = pd.read_csv(synonym_file)
                synonym_overrides_df.columns = synonym_overrides_df.columns.str.lower()
                if not {"term", "group_label"}.issubset(synonym_overrides_df.columns):
                    st.warning("⚠️ Synonym file needs `term` and `group_label` columns.")
                    synonym_overrides_df = None
                else:
                    st.success(f"✅ Loaded {len(synonym_overrides_df)} synonym mappings")

            st.success(f"✅ Loaded {len(checkouts_df)} checkout records and "
                       f"{len(recommendations_df)} recommendations")

            with st.expander("📋 Preview Data"):
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.write("**Checkouts:**")
                    st.dataframe(checkouts_df.head())
                with pc2:
                    st.write("**Recommendations:**")
                    st.dataframe(recommendations_df.head())

            # --- (Collection Insights panel removed — see Collection Profiler instead) ---

            # Scoring configuration
            st.subheader("⚙️ Step 2: Configure Scoring Weights")
            _fac_default = 0.15 if faculty_df is not None else 0.0
            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1:
                subject_weight = st.slider("Subject Similarity", 0.0, 1.0,
                                            0.45 if faculty_df else 0.5, 0.05, key="rec_sw")
            with wc2:
                lc_weight = st.slider("LC Classification", 0.0, 1.0,
                                       0.25 if faculty_df else 0.3, 0.05, key="rec_lw")
            with wc3:
                author_weight = st.slider("Author Popularity", 0.0, 1.0,
                                           0.15 if faculty_df else 0.2, 0.05, key="rec_aw")
            with wc4:
                faculty_weight = st.slider("Faculty Interest", 0.0, 1.0, _fac_default, 0.05,
                                            disabled=(faculty_df is None), key="rec_fw")

            total_weight = subject_weight + lc_weight + author_weight + faculty_weight
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Adjust so they total 1.0.")
            else:
                st.success(f"✅ Weights sum to {total_weight:.2f}")

            for key in ("rec_results", "rec_checkouts_scored", "rec_recs_scored",
                        "rec_faculty_scored", "rec_weights"):
                if key not in st.session_state:
                    st.session_state[key] = None

            if st.button("🚀 Score Recommendations", type="primary", key="rec_score_btn"):
                with st.spinner("Analyzing..."):
                    _stemmer = SnowballStemmer("english")
                    syn_map = build_synonym_map(_stemmer, synonym_overrides_df)
                    scorer = RecommendationScorer(checkouts_df, synonym_map=syn_map)
                    _faculty_scorer = None
                    if faculty_df is not None and faculty_weight > 0:
                        _faculty_scorer = FacultyScorer(faculty_df, _stemmer, syn_map)
                    results_df = scorer.score_recommendations(
                        recommendations_df,
                        subject_weight=subject_weight, lc_weight=lc_weight,
                        author_weight=author_weight, faculty_weight=faculty_weight,
                        faculty_scorer=_faculty_scorer,
                    )
                st.success("✅ Analysis complete!")
                st.session_state["rec_results"] = results_df
                st.session_state["rec_checkouts_scored"] = checkouts_df
                st.session_state["rec_recs_scored"] = recommendations_df
                st.session_state["rec_faculty_scored"] = faculty_df
                st.session_state["rec_weights"] = {
                    "subject": subject_weight, "lc": lc_weight,
                    "author": author_weight, "faculty": faculty_weight,
                }

            # Results display
            if st.session_state["rec_results"] is not None:
                results_df = st.session_state["rec_results"]
                st.subheader("📊 Step 3: Review Results")

                tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs([
                    "📊 Scored Recommendations", "📈 Score Distribution",
                    "🏷️ Subject Analysis", "🎓 Faculty Analysis"
                ])

                with tab_r1:
                    high_p = results_df[results_df["likelihood_score"] >= 70]
                    med_p = results_df[(results_df["likelihood_score"] >= 40) &
                                        (results_df["likelihood_score"] < 70)]
                    low_p = results_df[results_df["likelihood_score"] < 40]
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Total Scored", len(results_df))
                    tc2.metric("🟢 High (70+)", len(high_p))
                    tc3.metric("🟡 Medium (40-69)", len(med_p))
                    tc4.metric("🔴 Low (<40)", len(low_p))

                    search = st.text_input("Search by title or author", "", key="rec_search")
                    min_score = st.slider("Minimum score", 0, 100, 0, key="rec_min")
                    filtered = results_df.copy()
                    if search:
                        mask = (filtered["title"].str.contains(search, case=False, na=False) |
                                filtered.get("author", pd.Series(dtype=str))
                                .str.contains(search, case=False, na=False))
                        filtered = filtered[mask]
                    filtered = filtered[filtered["likelihood_score"] >= min_score]

                    def get_priority(s):
                        if s >= 70: return "🟢 High"
                        if s >= 40: return "🟡 Medium"
                        return "🔴 Low"

                    display = filtered.copy()
                    display["Priority"] = display["likelihood_score"].apply(get_priority)
                    pcols = ["Priority", "title", "author", "likelihood_score",
                             "similarity_score", "checkout_volume_score",
                             "author_popularity_score", "faculty_interest_score", "matched_faculty"]
                    others = [c for c in display.columns if c not in pcols]
                    display = display[[c for c in pcols if c in display.columns] + others]
                    st.dataframe(display, use_container_width=True, height=600)

                with tab_r2:
                    scores = results_df["likelihood_score"]
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=scores, nbinsx=20, marker_color="#285C4D"))
                    fig_hist.add_vline(x=70, line_dash="dash", line_color="#2ecc71",
                                       annotation_text="High (70)")
                    fig_hist.add_vline(x=40, line_dash="dash", line_color="#f39c12",
                                       annotation_text="Medium (40)")
                    fig_hist.update_layout(title="Score Distribution",
                                           xaxis_title="Score", yaxis_title="Count",
                                           height=400, showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Mean", f"{scores.mean():.1f}")
                    sc2.metric("Median", f"{scores.median():.1f}")
                    sc3.metric("Std Dev", f"{scores.std():.1f}")

                with tab_r3:
                    co_subj = extract_all_subjects(st.session_state["rec_checkouts_scored"])
                    rec_subj = extract_all_subjects(st.session_state["rec_recs_scored"])
                    sa1, sa2, sa3 = st.columns(3)
                    sa1.metric("Checkout Subjects", len(co_subj))
                    sa2.metric("Recommendation Subjects", len(rec_subj))
                    overlap = len(set(co_subj) & set(rec_subj))
                    sa3.metric("Common Subjects", overlap)

                    common = {s: {"co": co_subj[s], "rec": rec_subj[s]}
                              for s in set(co_subj) & set(rec_subj)}
                    if common:
                        cdf = pd.DataFrame([
                            {"Subject": s, "In Checkouts": d["co"], "In Recommendations": d["rec"],
                             "Total": d["co"] + d["rec"]}
                            for s, d in common.items()
                        ]).sort_values("Total", ascending=False)
                        st.dataframe(cdf.head(30), use_container_width=True, height=400)

                    gap_subj = {k: v for k, v in co_subj.items() if k not in rec_subj and v >= 2}
                    if gap_subj:
                        st.subheader("🔍 Recommendation Gaps")
                        st.markdown("High-circulation subjects missing from your recommendations list:")
                        gdf = pd.DataFrame([
                            {"Subject": s, "Checkout Occurrences": c}
                            for s, c in sorted(gap_subj.items(), key=lambda x: -x[1])[:30]
                        ])
                        st.dataframe(gdf, use_container_width=True, height=300)

                with tab_r4:
                    fac_scored = st.session_state.get("rec_faculty_scored")
                    if fac_scored is None:
                        st.info("No faculty file uploaded. Upload one and re-score to see this analysis.")
                    elif ("faculty_interest_score" not in results_df.columns
                          or results_df["faculty_interest_score"].sum() == 0):
                        st.warning("Faculty scores are zero — set faculty weight > 0 and re-score.")
                    else:
                        fac_results = results_df[results_df["matched_faculty"].str.strip() != ""].copy()
                        fc1, fc2 = st.columns(2)
                        fc1.metric("Faculty Members", len(fac_scored))
                        fc2.metric("Matched Recommendations", len(fac_results))
                        if len(fac_results) > 0:
                            st.dataframe(
                                fac_results.sort_values("faculty_interest_score", ascending=False)
                                .head(20)[["title", "author", "likelihood_score",
                                           "faculty_interest_score", "matched_faculty"]],
                                use_container_width=True, height=400
                            )

                # Downloads
                st.subheader("💾 Step 4: Download Results")
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.download_button("📥 Full Results (CSV)",
                                       results_df.to_csv(index=False),
                                       "recommendations_scored.csv", "text/csv",
                                       key="rec_dl_full")
                with dc2:
                    high_csv = results_df[results_df["likelihood_score"] >= 70].to_csv(index=False)
                    st.download_button("🟢 High Priority Only", high_csv,
                                       "recommendations_high_priority.csv", "text/csv",
                                       key="rec_dl_high")
                with dc3:
                    st.download_button("📄 Report (TXT)", generate_report(results_df),
                                       "recommendation_report.txt", "text/plain",
                                       key="rec_dl_txt")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Check that your CSV files have the required columns.")
    else:
        st.info("👆 Upload both a checkouts file and a recommendations file to begin.")

    # Sidebar instructions
    with st.sidebar:
        st.markdown("---")
        st.subheader("📖 Scorer Instructions")
        st.markdown("""
        1. Upload **checkouts CSV** *(required)*
        2. Upload **recommendations CSV** *(required)*
        3. Optionally add **faculty** and **synonym** CSVs
        4. Adjust scoring weights
        5. Click **Score Recommendations**

        **Scores:** 🟢 70+ High · 🟡 40-69 Medium · 🔴 <40 Low
        """)


# =====================================================================
# HOME PAGE & MAIN NAVIGATION
# =====================================================================

def page_home():
    st.title("📚 Library Collection Dashboard")
    st.markdown(
        "A decision-support suite for collection assessment, subscription management, "
        "and acquisition prioritization at Howard-Tilton Memorial Library."
    )
    st.markdown(
        "Each tool answers one question. Pick based on what you need to decide."
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="tool-card">
            <h3>🗺️ Collection Profiler</h3>
            <p><em>What does our collection look like?</em></p>
            <p>LC sunburst, treemap, subject word cloud, gap analysis.
            Map disciplinary strengths across 1M+ records.</p>
            <hr>
            <p><strong>Use for:</strong></p>
            <ul>
                <li>Baseline & accreditation reports</li>
                <li>Liaison prep</li>
                <li>Budget justifications</li>
                <li>Weeding area selection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="tool-card">
            <h3>📈 Usage & Subscription Analyzer</h3>
            <p><em>What's being used and what isn't?</em></p>
            <p>Title-level usage for COUNTER e-resources and print circulation.
            Cancellation candidates, publisher rollups, trends.</p>
            <hr>
            <p><strong>Use for:</strong></p>
            <ul>
                <li>Annual database renewals</li>
                <li>Print weeding by circulation</li>
                <li>Package value evaluation</li>
                <li>Faculty cancellation conversations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="tool-card">
            <h3>📊 Acquisition Recommendation Scorer</h3>
            <p><em>What should we buy next?</em></p>
            <p>Score candidate books against checkout history +
            faculty research interests. Subject, LC, author, faculty.</p>
            <hr>
            <p><strong>Use for:</strong></p>
            <ul>
                <li>Vendor slip list triage</li>
                <li>Approval-plan exceptions</li>
                <li>Faculty request review</li>
                <li>DDA flip decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧭 Quick decision guide")
    st.markdown("""
    | You need to… | Use |
    |---|---|
    | Show what the collection covers (or doesn't) | **Collection Profiler** |
    | Decide which databases to renew or cancel | **Usage & Subscription Analyzer** → COUNTER |
    | Pick books to weed from the stacks | **Usage & Subscription Analyzer** → Print |
    | Prioritize purchases from a vendor list | **Recommendation Scorer** |
    | Match new books to specific faculty research | **Recommendation Scorer** (with faculty file) |
    | Compare what we own to what's being checked out | **Collection Profiler** (with usage column) |
    | Find high-use areas missing from the catalog | **Recommendation Scorer** → Subject Analysis tab |
    """)

    st.markdown("---")
    with st.expander("ℹ️ About this dashboard"):
        st.markdown("""
        **Version 2.0** — consolidated from four earlier Streamlit apps:
        - *Word Cloud Generator* (now a view option inside the Collection Profiler)
        - *Collection Strengths Analyzer* (now the Collection Profiler backbone)
        - *EBSCO Subscription Dashboard* (now the Usage Analyzer → COUNTER mode)
        - *Book Recommendation Scorer* (now the Acquisition Recommendation Scorer)

        **Design principles:**
        - Each tool answers a different decision question
        - Shared utilities (LC parsing, text normalization, CSV loading) live once
        - Memory-optimized for large catalog exports

        Built with Streamlit. Tulane color palette: `#285C4D` (green), `#71C5E8` (blue).
        Fonts: Source Serif 4 / DM Sans.
        """)


def main():
    with st.sidebar:
        st.title("📚 Collection Dashboard")
        st.markdown("*Howard-Tilton Memorial Library*")
        st.markdown("---")
        page = st.radio(
            "Select a tool:",
            ["🏠 Home",
             "🗺️ Collection Profiler",
             "📈 Usage & Subscription Analyzer",
             "📊 Acquisition Recommendation Scorer"],
            index=0,
            key="nav"
        )
        st.markdown("---")

    if page == "🏠 Home":
        page_home()
    elif page == "🗺️ Collection Profiler":
        page_collection_profiler()
    elif page == "📈 Usage & Subscription Analyzer":
        page_usage_analyzer()
    elif page == "📊 Acquisition Recommendation Scorer":
        page_recommendation_scorer()

    _footer()


if __name__ == "__main__":
    main()
