"""
Library Collection Dashboard (slim edition)
===========================================
A unified Streamlit application bundling four collection decision-support tools:

  1. Collection Profiler — "What does our collection look like?"
     Holdings structure only:
       • LC Analysis — sunburst, treemap, LC × subject heatmap, sub-class range
         distribution. Feeds collection assessment and accreditation work.
       • Subject Term Analysis — top subjects, word cloud, title-keyword n-grams.
         Feeds policy revision, liaison conversations, marketing planning.
     Usage-driven views (Coverage-vs-Use, top titles, gap, weeding triage) moved
     to Use Analysis as of v2.7.

  2. Use Analysis — "What's getting used, and is it worth keeping?"
     One tool for all usage-driven work, branching by data type:
       • Print circulation (subject + usage) — profiler engine with usage on:
         Coverage-vs-Use, top titles, gap-vs-use, weeding triage.
       • Electronic / COUNTER 5 — formal COUNTER reports (TR/TR_J3/TR_B1/DR/PR/IR)
         via the COUNTER reader: cost-per-use, dead weight, monthly trends.
       • Electronic / other usage — any title-level usage export.
     The print and other-usage branches expect the synced explicit-zero master
     from the Zero-Use Identifier so unused titles count as 0.

  3. Zero-Use Identifier — "What do we own that isn't being used?"
     Holdings vs. usage comparison with a multi-identifier matching cascade
     (ISBN, ISSN, DOI, OCLC, title+author fallback). Retains subject metadata and
     emits two outputs: the zero-use title list, and the explicit-zero master
     (every title with a numeric use value, 0 where unused) that feeds Use Analysis.

  4. Overlap & Uniqueness — "What's unique to each database?"
     Reads an e-journal coverage / A-to-Z export and classifies every title per
     database as sole source, unique coverage, or redundant, using day-resolution
     interval math so date coverage (not just title name) drives the picture.

v2.7 (slim) — Full cut: usage analysis consolidated into a single Use Analysis
       tool (print circulation, COUNTER 5, and other usage data). The Collection
       Profiler is now structure-only; the standalone COUNTER Analyzer is retired
       as a top-level tool (its reader is the Use Analysis COUNTER branch). The
       Zero-Use Identifier retains subjects and emits two outputs only.
v2.6 (slim) — Added the Overlap & Uniqueness tool: reads an e-journal coverage /
       A-to-Z export and classifies every title per database as sole source,
       unique coverage, or redundant, using day-resolution interval math so
       date coverage (not just title name) drives the cancellation picture.
v2.5 (slim) — Added inline "show the records behind this" drill-downs across the
       Profiler's coverage-vs-use views, range distribution, and subject bars,
       with in-place usage/year filtering, sorting, and CSV export.
v2.4 (slim) — Acquisition Recommendation Scorer extracted to its own standalone
       app (recommender_app.py). NLTK is no longer a runtime dependency.
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

# Excel support — pandas uses openpyxl for .xlsx and xlrd for .xls.
# Both are optional dependencies; CSV path always works.
try:
    import openpyxl  # noqa: F401
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

try:
    import xlrd  # noqa: F401
    XLS_AVAILABLE = True
except ImportError:
    XLS_AVAILABLE = False


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
    /* deprecated — formerly wrapped file uploaders; kept only to avoid breaking any cached HTML */
    display: none;
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

# LC subclass map — main letter → {two-letter subclass code → human label}
# Sourced from the Library of Congress Classification Outline
# (https://www.loc.gov/aba/cataloging/classification/lcco/) and the LC's
# free per-class PDF schedules. Covers all 21 main classes.
#
# Coverage scope: the two-letter (alpha) subclasses only. The numerical
# ranges below those (e.g., HQ 1000–1999) are not represented here because
# the dashboard's matching only inspects the leading letters of a call
# number — a richer breakdown isn't useful unless we change the parser.
#
# A few classes have notable nuances reflected here:
#   - Class K (Law): the largest schedule, with subclasses spanning regions
#     and religious/jurisdictional systems (KBM/KBP/KBR/KBU, KD-KDK, KJ-KKZ,
#     KL-KWX). Three-letter codes like KBM, KDC, KDE, KDK are included where
#     they're commonly seen in practice.
#   - Class P (Language & Literature): includes both language subclasses
#     (P-PM) and literature subclasses (PN-PZ).
#   - Class E and Class F have no alpha subclasses in the standard schedule
#     (they use numeric ranges only), so they're omitted from this dict.
LC_SUBCLASSES = {
    'A': {
        'A': 'General Works (General)',
        'AC': 'Collections, Series, Collected Works',
        'AE': 'Encyclopedias',
        'AG': 'Dictionaries & General Reference',
        'AI': 'Indexes',
        'AM': 'Museums, Collectors & Collecting',
        'AN': 'Newspapers',
        'AP': 'Periodicals',
        'AS': 'Academies & Learned Societies',
        'AY': 'Yearbooks, Almanacs, Directories',
        'AZ': 'History of Scholarship & Learning',
    },
    'B': {
        'B': 'Philosophy (General)',
        'BC': 'Logic',
        'BD': 'Speculative Philosophy',
        'BF': 'Psychology',
        'BH': 'Aesthetics',
        'BJ': 'Ethics',
        'BL': 'Religions, Mythology, Rationalism',
        'BM': 'Judaism',
        'BP': 'Islam, Bahai',
        'BQ': 'Buddhism',
        'BR': 'Christianity',
        'BS': 'The Bible',
        'BT': 'Doctrinal Theology',
        'BV': 'Practical Theology',
        'BX': 'Christian Denominations',
    },
    'C': {
        'C': 'Auxiliary Sciences of History (General)',
        'CB': 'History of Civilization',
        'CC': 'Archaeology',
        'CD': 'Diplomatics, Archives, Seals',
        'CE': 'Technical Chronology, Calendar',
        'CJ': 'Numismatics',
        'CN': 'Inscriptions, Epigraphy',
        'CR': 'Heraldry',
        'CS': 'Genealogy',
        'CT': 'Biography',
    },
    'D': {
        'D': 'World History (General)',
        'DA': 'History of Great Britain',
        'DAW': 'History of Central Europe',
        'DB': 'History of Austria, Hungary, Czechia, Slovakia',
        'DC': 'History of France, Andorra, Monaco',
        'DD': 'History of Germany',
        'DE': 'History of the Greco-Roman World',
        'DF': 'History of Greece',
        'DG': 'History of Italy, Vatican, Malta',
        'DH': 'History of Low Countries, Benelux',
        'DJ': 'History of Netherlands',
        'DJK': 'History of Eastern Europe (General)',
        'DK': 'History of Russia, Soviet Union, Former Soviet Republics',
        'DL': 'History of Northern Europe, Scandinavia',
        'DP': 'History of Iberian Peninsula, Spain, Portugal',
        'DQ': 'History of Switzerland',
        'DR': 'History of Balkan Peninsula',
        'DS': 'History of Asia',
        'DT': 'History of Africa',
        'DU': 'History of Oceania, Pacific Area',
        'DX': 'History of Romani People',
    },
    # E and F have no alpha subclasses in the standard LC schedule —
    # they use numeric ranges only (e.g., E11–143, F1–975). Not represented.
    'G': {
        'G': 'Geography (General), Atlases, Maps',
        'GA': 'Mathematical Geography, Cartography',
        'GB': 'Physical Geography',
        'GC': 'Oceanography',
        'GE': 'Environmental Sciences',
        'GF': 'Human Ecology, Anthropogeography',
        'GN': 'Anthropology',
        'GR': 'Folklore',
        'GT': 'Manners & Customs',
        'GV': 'Recreation, Leisure',
    },
    'H': {
        'H': 'Social Sciences (General)',
        'HA': 'Statistics',
        'HB': 'Economic Theory, Demography',
        'HC': 'Economic History & Conditions',
        'HD': 'Industries, Land Use, Labor',
        'HE': 'Transportation & Communications',
        'HF': 'Commerce',
        'HG': 'Finance',
        'HJ': 'Public Finance',
        'HM': 'Sociology (General)',
        'HN': 'Social History & Conditions',
        'HQ': 'The Family, Marriage, Women, Sexuality',
        'HS': 'Societies, Secret, Benevolent, etc.',
        'HT': 'Communities, Classes, Races',
        'HV': 'Social Pathology, Criminology, Welfare',
        'HX': 'Socialism, Communism, Anarchism',
    },
    'J': {
        'J': 'General Legislative & Executive Papers',
        'JA': 'Political Science (General)',
        'JC': 'Political Theory',
        'JF': 'Political Institutions & Public Administration (General)',
        'JJ': 'Political Institutions — North America',
        'JK': 'Political Institutions — United States',
        'JL': 'Political Institutions — Canada, Latin America',
        'JN': 'Political Institutions — Europe',
        'JQ': 'Political Institutions — Asia, Africa, Pacific',
        'JS': 'Local Government, Municipal Government',
        'JV': 'Colonies & Colonization, Migration',
        'JX': 'International Law (obsolete; see JZ & KZ)',
        'JZ': 'International Relations',
    },
    'K': {
        'K': 'Law in General, Comparative & Uniform Law, Jurisprudence',
        'KB': 'Religious Law (General), Comparative',
        'KBM': 'Jewish Law',
        'KBP': 'Islamic Law',
        'KBR': 'History of Canon Law',
        'KBU': 'Law of the Roman Catholic Church, Holy See',
        'KD': 'Law of the United Kingdom & Ireland',
        'KDC': 'Law of Scotland',
        'KDE': 'Law of Northern Ireland',
        'KDG': 'Law of Isle of Man, Channel Islands',
        'KDK': 'Law of Ireland (Eire)',
        'KDZ': 'America, North America (General)',
        'KE': 'Law of Canada',
        'KF': 'Law of the United States',
        'KG': 'Law of Latin America, Mexico, Central America, West Indies',
        'KH': 'Law of South America',
        'KJ': 'Law of Europe (General)',
        'KJA': 'Roman Law',
        'KJC': 'Regional Comparative Law (Europe)',
        'KJE': 'Regional Organization & Integration (Europe)',
        'KJV': 'Law of France',
        'KK': 'Law of Germany',
        'KL': 'History of Law, Ancient Orient',
        'KLA': 'Law of Russia, Soviet Union',
        'KM': 'Law of Asia, Middle East, Southwest Asia',
        'KN': 'Law of South, Southeast & East Asia',
        'KP': 'Law of South, Southeast & East Asia (continued)',
        'KQ': 'Law of Africa',
        'KR': 'Law of Africa (continued)',
        'KS': 'Law of Africa (continued)',
        'KT': 'Law of Africa (continued)',
        'KU': 'Law of Pacific Area',
        'KV': 'Law of Pacific Area (continued)',
        'KW': 'Law of Pacific Area, Antarctica',
        'KZ': 'Law of Nations, International Law',
    },
    'L': {
        'L': 'Education (General)',
        'LA': 'History of Education',
        'LB': 'Theory & Practice of Education',
        'LC': 'Social Aspects of Education',
        'LD': 'Individual Educational Institutions — United States',
        'LE': 'Individual Educational Institutions — America (excluding US)',
        'LF': 'Individual Educational Institutions — Europe',
        'LG': 'Individual Educational Institutions — Asia, Africa, Oceania',
        'LH': 'College & School Magazines & Papers',
        'LJ': 'Student Fraternities & Societies, US',
        'LT': 'Textbooks',
    },
    'M': {
        'M': 'Music (Scores, Performance)',
        'ML': 'Literature on Music',
        'MT': 'Musical Instruction & Study',
    },
    'N': {
        'N': 'Visual Arts (General)',
        'NA': 'Architecture',
        'NB': 'Sculpture',
        'NC': 'Drawing, Design, Illustration',
        'ND': 'Painting',
        'NE': 'Print Media',
        'NK': 'Decorative Arts',
        'NX': 'Arts in General',
    },
    'P': {
        'P': 'Philology & Linguistics (General)',
        'PA': 'Greek & Latin Language & Literature',
        'PB': 'Modern Languages, Celtic Languages',
        'PC': 'Romance Languages',
        'PD': 'Germanic & Scandinavian Languages',
        'PE': 'English Language',
        'PF': 'West Germanic Languages',
        'PG': 'Slavic, Baltic, Albanian Languages',
        'PH': 'Uralic & Basque Languages',
        'PJ': 'Oriental Languages & Literatures',
        'PK': 'Indo-Iranian Languages & Literatures',
        'PL': 'Languages & Literatures of East Asia, Africa, Oceania',
        'PM': 'Hyperborean, Indigenous American & Artificial Languages',
        'PN': 'Literature (General)',
        'PQ': 'French, Italian, Spanish, Portuguese Literatures',
        'PR': 'English Literature',
        'PS': 'American Literature',
        'PT': 'German, Dutch, Scandinavian Literatures',
        'PZ': 'Fiction & Juvenile Belles Lettres',
    },
    'Q': {
        'Q': 'Science (General)',
        'QA': 'Mathematics (incl. Computer Science)',
        'QB': 'Astronomy',
        'QC': 'Physics',
        'QD': 'Chemistry',
        'QE': 'Geology',
        'QH': 'Natural History, Biology (General)',
        'QK': 'Botany',
        'QL': 'Zoology',
        'QM': 'Human Anatomy',
        'QP': 'Physiology',
        'QR': 'Microbiology',
    },
    'R': {
        'R': 'Medicine (General)',
        'RA': 'Public Aspects of Medicine',
        'RB': 'Pathology',
        'RC': 'Internal Medicine',
        'RD': 'Surgery',
        'RE': 'Ophthalmology',
        'RF': 'Otorhinolaryngology',
        'RG': 'Gynecology & Obstetrics',
        'RJ': 'Pediatrics',
        'RK': 'Dentistry',
        'RL': 'Dermatology',
        'RM': 'Therapeutics, Pharmacology',
        'RS': 'Pharmacy & Materia Medica',
        'RT': 'Nursing',
        'RV': 'Botanic, Thomsonian & Eclectic Medicine',
        'RX': 'Homeopathy',
        'RZ': 'Other Systems of Medicine (incl. Chiropractic)',
    },
    'S': {
        'S': 'Agriculture (General)',
        'SB': 'Plant Culture, Horticulture',
        'SD': 'Forestry',
        'SF': 'Animal Culture, Veterinary Medicine',
        'SH': 'Aquaculture, Fisheries, Angling',
        'SK': 'Hunting Sports',
    },
    'T': {
        'T': 'Technology (General)',
        'TA': 'Engineering (General), Civil Engineering',
        'TC': 'Hydraulic & Ocean Engineering',
        'TD': 'Environmental Technology, Sanitary Engineering',
        'TE': 'Highway Engineering, Roads & Pavements',
        'TF': 'Railroad Engineering & Operation',
        'TG': 'Bridge Engineering',
        'TH': 'Building Construction',
        'TJ': 'Mechanical Engineering & Machinery',
        'TK': 'Electrical & Nuclear Engineering, Electronics',
        'TL': 'Motor Vehicles, Aeronautics, Astronautics',
        'TN': 'Mining Engineering, Metallurgy',
        'TP': 'Chemical Technology',
        'TR': 'Photography',
        'TS': 'Manufactures',
        'TT': 'Handicrafts, Arts & Crafts',
        'TX': 'Home Economics',
    },
    'U': {
        'U': 'Military Science (General)',
        'UA': 'Armies: Organization, Description, Facilities',
        'UB': 'Military Administration',
        'UC': 'Maintenance & Transportation',
        'UD': 'Infantry',
        'UE': 'Cavalry, Armor',
        'UF': 'Artillery',
        'UG': 'Military Engineering, Air Forces, Air Warfare',
        'UH': 'Other Military Services (Medical, Welfare, etc.)',
    },
    'V': {
        'V': 'Naval Science (General)',
        'VA': 'Navies: Organization, Description, Facilities',
        'VB': 'Naval Administration',
        'VC': 'Naval Maintenance',
        'VD': 'Naval Seamen',
        'VE': 'Marines',
        'VF': 'Naval Ordnance',
        'VG': 'Minor Naval Services',
        'VK': 'Navigation, Merchant Marine',
        'VM': 'Naval Architecture, Shipbuilding, Marine Engineering',
    },
    'Z': {
        'Z': 'Bibliography, Library Science, Information Resources (General)',
        'ZA': 'Information Resources, Information Science',
    },
}


# LC numeric range catalog — subclass code → list of (start, end, label) tuples.
# Each tuple defines an LC-defined range within a two-letter subclass, drawn from
# the Library of Congress Classification Outline (https://www.loc.gov/aba/cataloging/classification/lcco/).
#
# Coverage scope: substantive but bounded. Covers the highest-traffic subclasses
# across humanities, social sciences, and STEM where library users most often
# need range-level granularity. Subclasses not listed fall back to bucketing
# by hundreds (e.g., "HX 100s") via _bucket_by_hundreds.
#
# Range semantics:
#   - start, end are inclusive
#   - end can be a float (HQ 1101-2030.7) — LC ranges sometimes use decimals
#   - ranges should not overlap within a subclass; if they do, the first match wins
#   - sort within each subclass list by `start` ascending for clarity
#
# To extend: add entries for additional subclasses, or refine existing ones.
# Curated against the LC outline; some ranges combine adjacent LC sub-ranges
# where the distinction is finer than the dashboard needs to surface.
LC_RANGES = {
    # ===== Class B — Philosophy, Psychology, Religion =====
    'B': [
        (1, 68, 'Philosophy (General)'),
        (69, 5739, 'History & systems of philosophy'),
    ],
    'BF': [
        (1, 1000, 'Psychology'),
        (1001, 1389, 'Parapsychology'),
        (1404, 2055, 'Occult sciences'),
    ],
    'BJ': [
        (1, 1725, 'Ethics'),
        (1801, 2195, 'Social usages, etiquette'),
    ],
    'BL': [
        (1, 50, 'Religion (General)'),
        (51, 65, 'Philosophy of religion'),
        (70, 980, 'History & principles of religions'),
        (1000, 2370, 'Asian, Indian, & Iranian religions'),
        (2390, 2630, 'African, Pacific, Indigenous religions'),
    ],
    'BR': [
        (1, 1725, 'Christianity'),
    ],
    'BS': [
        (1, 2970, 'The Bible'),
    ],
    'BT': [
        (1, 1480, 'Doctrinal theology'),
    ],
    'BV': [
        (1, 5099, 'Practical theology'),
    ],
    'BX': [
        (1, 4795, 'Christian denominations'),
        (4800, 9999, 'Protestant denominations'),
    ],

    # ===== Class C — Auxiliary Sciences of History =====
    'C': [
        (1, 51, 'Auxiliary sciences of history (general)'),
    ],
    'CB': [
        (3, 482, 'History of civilization'),
        (156, 161, 'Ancient civilizations'),
        (203, 281, 'Civilization & medieval period'),
        (351, 482, 'Modern civilization'),
    ],
    'CC': [
        (1, 960, 'Archaeology'),
    ],
    'CD': [
        (1, 6471, 'Diplomatics, archives, seals'),
        (921, 4280, 'Archives'),
    ],
    'CE': [
        (1, 97, 'Technical chronology, calendar'),
    ],
    'CJ': [
        (1, 6661, 'Numismatics (coins, medals)'),
    ],
    'CN': [
        (1, 1355, 'Inscriptions, epigraphy'),
    ],
    'CR': [
        (1, 6305, 'Heraldry'),
    ],
    'CS': [
        (1, 3090, 'Genealogy'),
    ],
    'CT': [
        (21, 9999, 'Biography'),
        (21, 22, 'Biography as an art or literary form'),
        (100, 3150, 'General collective biography'),
        (3200, 3830, 'Biography of women (collective)'),
        (3990, 9999, 'Biography of specific groups or classes'),
    ],

    # ===== Class D — World History =====
    'D': [
        (1, 2027, 'World history (general)'),
        (51, 95, 'Ancient history'),
        (101, 199, 'Medieval history'),
        (200, 475, 'Modern history (general)'),
        (501, 680, 'World War I (1914–1918)'),
        (731, 838, 'World War II (1939–1945)'),
        (839, 860, 'Post-war history (1945–1989)'),
        (861, 2027, 'Post-Cold War period (1989–)'),
    ],
    'DA': [
        (1, 990, 'Great Britain (general)'),
        (700, 745, 'Wales'),
        (750, 890, 'Scotland'),
        (900, 995, 'Ireland'),
    ],
    'DB': [
        (1, 3150, 'Austria, Hungary, Czechia, Slovakia'),
        (2000, 3150, 'Hungary, Czech Republic, Slovakia'),
    ],
    'DC': [
        (1, 947, 'France'),
        (33, 59.8, 'France: pre-1815'),
        (60, 424, 'France: 1815–present'),
        (601, 800, 'Andorra, Monaco'),
        (921, 947, 'France: regional history'),
    ],
    'DD': [
        (1, 905, 'Germany'),
        (256.5, 257.4, 'Holy Roman Empire'),
        (258, 290, 'Germany: 1918–1945 (Weimar, Nazi era)'),
        (290, 905, 'Germany: 1945–present (incl. divided & reunified)'),
    ],
    'DE': [
        (1, 100, 'Greco-Roman world (general)'),
    ],
    'DF': [
        (10, 951, 'Greece'),
        (10, 289, 'Ancient Greece'),
        (501, 649, 'Byzantine Empire'),
        (701, 951, 'Modern Greece'),
    ],
    'DG': [
        (11, 999, 'Italy'),
        (11, 365, 'Ancient Italy & Rome'),
        (401, 583, 'Medieval & Renaissance Italy'),
        (601, 875, 'Modern Italy'),
    ],
    'DJ': [
        (1, 401, 'Netherlands'),
    ],
    'DJK': [
        (1, 77, 'Eastern Europe (general)'),
    ],
    'DK': [
        (1, 949, 'Russia, Soviet Union, former Soviet republics'),
        (1, 274, 'Russia: pre-1917'),
        (245, 274, 'Russian Revolution'),
        (266, 290, 'Soviet Union (USSR), 1917–1991'),
        (501, 949, 'Russia: 1991–present + former republics'),
    ],
    'DL': [
        (1, 1180, 'Northern Europe, Scandinavia'),
    ],
    'DP': [
        (1, 402, 'Spain & Portugal'),
        (1, 272, 'Spain'),
        (501, 802, 'Portugal'),
    ],
    'DR': [
        (1, 2285, 'Balkan Peninsula'),
        (401, 741, 'Turkey'),
        (901, 2285, 'Albania, Bulgaria, Romania, Yugoslavia (former), etc.'),
    ],
    'DS': [
        (1, 937, 'Asia'),
        (1, 41, 'Asia (general)'),
        (51, 95.9, 'Middle East'),
        (101, 151, 'Israel, Palestine'),
        (251, 326, 'Iran (Persia)'),
        (327, 329.4, 'Central Asia'),
        (331, 349.9, 'Southern Asia (general)'),
        (350, 375, 'Afghanistan'),
        (376, 392, 'Pakistan'),
        (401, 486, 'India'),
        (488, 490, 'Sri Lanka'),
        (491, 492, 'Bhutan, Nepal'),
        (501, 526.7, 'East Asia (general)'),
        (701, 799.9, 'China'),
        (798, 799.9, 'Hong Kong, Macao, Taiwan'),
        (801, 897, 'Japan'),
        (901, 937, 'Korea'),
    ],
    'DT': [
        (1, 3415, 'Africa'),
        (1, 159.9, 'Africa (general & North Africa)'),
        (43, 154, 'Egypt'),
        (160, 177, 'Libya'),
        (181, 346, 'Maghreb (Tunisia, Algeria, Morocco)'),
        (348, 363, 'West & Central Africa (general)'),
        (470, 671, 'West Africa'),
        (777, 1465, 'Eastern, Southern Africa'),
        (1501, 2405, 'Southern Africa'),
        (2421, 2999, 'Madagascar & adjacent islands'),
        (3001, 3415, 'Other African regions'),
    ],
    'DU': [
        (1, 950, 'Oceania (Australia, New Zealand, Pacific)'),
        (80, 398, 'Australia'),
        (400, 430, 'New Zealand'),
        (490, 950, 'Pacific islands'),
    ],
    'DX': [
        (101, 301, 'Romanies (Gypsies)'),
    ],

    # ===== Class E — US History =====
    'E': [
        (11, 143, 'America (general); pre-Columbian; discovery'),
        (151, 909, 'United States (general)'),
        (151, 169, 'United States (general history)'),
        (171, 183.9, 'US history by period (general)'),
        (184, 200, 'Elements in the population'),
        (185, 185.98, 'African Americans'),
        (186, 199, 'Indigenous peoples of the Americas'),
        (201, 298, 'Colonial period (1607–1775)'),
        (300, 453, 'Revolution & Confederation (1775–1789)'),
        (456, 655, 'Civil War period (1861–1865)'),
        (660, 738, 'Late 19th century (1865–1900)'),
        (740, 837.7, '20th century (1900–2000)'),
        (838, 909, '21st century (2001–)'),
    ],

    # ===== Class F — History of the Americas =====
    'F': [
        (1, 975, 'United States local history (Northern, NE, Mid-Atlantic, Mid-West, South)'),
        (1, 15, 'New England'),
        (16, 215, 'Northeastern states (NY, NJ, PA, etc.)'),
        (221, 580, 'Southern states'),
        (586, 705, 'Mid-Western states'),
        (721, 975, 'Western states'),
        (1001, 1145.2, 'Canada'),
        (1170, 1170, 'Bermuda, Atlantic islands'),
        (1201, 1392, 'Mexico'),
        (1401, 3799, 'Latin America (general)'),
        (1401, 1419, 'Latin America (general)'),
        (1421, 1577, 'Central America'),
        (1601, 2151, 'West Indies (Caribbean)'),
        (2155, 2191, 'Guianas'),
        (2201, 2659, 'South America (general & Venezuela, Colombia)'),
        (2661, 2799, 'Brazil'),
        (2801, 3021, 'Argentina'),
        (3031, 3091, 'Uruguay'),
        (3101, 3201, 'Paraguay'),
        (3201, 3359, 'Bolivia'),
        (3401, 3619, 'Chile'),
        (3701, 3799, 'Peru'),
    ],

    # ===== Class G — Geography, Anthropology, Recreation =====
    'G': [
        (1, 922, 'Geography (general), atlases, maps'),
    ],
    'GA': [
        (1, 1776, 'Mathematical geography, cartography'),
    ],
    'GB': [
        (3, 5030, 'Physical geography'),
        (400, 649, 'Geomorphology'),
        (651, 2998, 'Hydrology, water resources'),
        (5000, 5030, 'Natural disasters'),
    ],
    'GC': [
        (1, 1581, 'Oceanography'),
    ],
    'GE': [
        (1, 350, 'Environmental sciences & studies'),
        (170, 190, 'Environmental policy'),
        (195, 199, 'Environmental management & sustainability'),
        (300, 350, 'Environmental ethics & justice'),
    ],
    'GF': [
        (1, 900, 'Human ecology, anthropogeography'),
    ],
    'GN': [
        (1, 890, 'Anthropology'),
        (49, 296, 'Physical anthropology'),
        (301, 674, 'Ethnology, social & cultural anthropology'),
        (700, 890, 'Prehistoric archaeology'),
    ],
    'GR': [
        (1, 950, 'Folklore'),
    ],
    'GT': [
        (1, 7070, 'Manners & customs (general)'),
        (3400, 5090, 'Customs relative to private life'),
        (5220, 7070, 'Customs relative to public & social life'),
    ],
    'GV': [
        (1, 1860, 'Recreation, leisure, sports'),
        (557, 1198.995, 'Sports (general & by sport)'),
        (1201, 1570, 'Games & amusements'),
        (1580, 1799, 'Dance'),
    ],

    # ===== Class H — Social Sciences =====
    'HB': [
        (1, 3840, 'Economic theory & demography'),
    ],
    'HC': [
        (10, 1085, 'Economic history & conditions'),
    ],
    'HD': [
        (1, 1130, 'Industries, land use, labor (general)'),
        (1131, 1395, 'Land tenure & agrarian reform'),
        (1401, 2210, 'Agricultural economics'),
        (2321, 4730, 'Industry'),
        (4801, 8943, 'Labor & class struggles'),
        (9000, 9999, 'Special industries'),
    ],
    'HF': [
        (1, 6182, 'Commerce'),
        (5001, 6182, 'Business & marketing'),
    ],
    'HG': [
        (1, 9999, 'Finance, money, banking'),
    ],
    'HM': [
        (1, 1281, 'Sociology (general)'),
    ],
    'HN': [
        (1, 995, 'Social history & conditions'),
    ],
    'HQ': [
        (1, 11, 'The family. Marriage. Women (general)'),
        (12, 449, 'Sexual life & sexuality'),
        (450, 472, 'Erotica'),
        (503, 1064, 'Family, marriage, home'),
        (1075, 1090.7, 'Sex role'),
        (1101, 2030.7, 'Women, feminism, women\'s studies'),
        (2035, 2039, 'Life skills'),
    ],
    'HT': [
        (51, 100, 'Communities (general)'),
        (101, 395, 'Urban sociology'),
        (401, 485, 'Rural sociology'),
        (601, 1445, 'Classes & class structure'),
        (1501, 1595, 'Races'),
    ],
    'HV': [
        (1, 4959, 'Social pathology, social work'),
        (5001, 5840, 'Substance abuse'),
        (6001, 7220.5, 'Criminology'),
        (7231, 9920.5, 'Criminal justice administration'),
    ],
    'HX': [
        (1, 970.7, 'Socialism, communism, anarchism'),
    ],

    # ===== Class J — Political Science =====
    'JA': [
        (1, 92, 'Political science (general)'),
    ],
    'JC': [
        (11, 605, 'Political theory'),
    ],
    'JF': [
        (20, 1177, 'Comparative government'),
        (1338, 2112, 'Public administration'),
    ],
    'JK': [
        (1, 9993, 'United States political institutions'),
    ],
    'JL': [
        (1, 3899, 'Americas (outside US)'),
    ],
    'JN': [
        (1, 9689, 'Europe'),
    ],
    'JQ': [
        (1, 6651, 'Asia, Africa, Australia, Oceania'),
    ],
    'JS': [
        (3, 8500, 'Local government & municipal government'),
    ],
    'JV': [
        (1, 9480, 'Colonies, colonization, emigration, immigration'),
    ],
    'JZ': [
        (5, 6530, 'International relations'),
    ],

    # ===== Class K — Law =====
    # The largest LC schedule. Subclasses cover both legal systems
    # (KB religious, KD UK, KE Canada, KF US federal, etc.) and regions.
    'K': [
        (1, 7720, 'Law (general); comparative & uniform law'),
        (370, 487, 'Comparative law'),
        (500, 5582, 'Jurisprudence & philosophy of law'),
    ],
    'KB': [
        (1, 4855, 'Religious law (general)'),
    ],
    'KBM': [
        (1, 4855, 'Jewish law'),
    ],
    'KBP': [
        (1, 4855, 'Islamic law'),
    ],
    'KBR': [
        (1, 1300, 'History of canon law'),
    ],
    'KBU': [
        (1, 4855, 'Law of the Roman Catholic Church'),
    ],
    'KD': [
        (51, 9684, 'United Kingdom & Ireland'),
        (51, 600, 'UK: sources & legal history'),
        (640, 7990, 'UK: law'),
        (8001, 9684, 'Ireland'),
    ],
    'KDC': [
        (1, 990, 'Scotland'),
    ],
    'KDE': [
        (1, 990, 'Northern Ireland'),
    ],
    'KDK': [
        (1, 9990, 'Ireland (Republic)'),
    ],
    'KDZ': [
        (1, 4999, 'America. North America'),
    ],
    'KE': [
        (1, 9450, 'Canada'),
    ],
    'KF': [
        (1, 9827, 'United States federal law'),
        (101, 130, 'Bibliography'),
        (140, 246, 'Legislative documents'),
        (4101, 4500, 'Constitutional law'),
        (4501, 4595, 'States (general)'),
        (4651, 4945, 'Civil rights'),
        (5050, 5455, 'Government & public administration'),
        (8700, 9050, 'Criminal procedure'),
        (8701, 9075, 'Criminal law'),
        (9201, 9479, 'Civil procedure & courts'),
        (9601, 9764, 'Procedure'),
        (9750, 9827, 'Criminal law (special topics)'),
    ],
    'KFA': [
        (1, 1000, 'US state law: Alabama, Alaska, Arizona, Arkansas'),
    ],
    'KFC': [
        (1, 1000, 'US state law: California, Colorado, Connecticut'),
    ],
    'KFD': [
        (1, 999, 'US state law: Delaware'),
    ],
    'KFF': [
        (1, 999, 'US state law: Florida'),
    ],
    'KFG': [
        (1, 999, 'US state law: Georgia'),
    ],
    'KFH': [
        (1, 999, 'US state law: Hawaii'),
    ],
    'KFI': [
        (1, 9999, 'US state law: Idaho, Illinois, Indiana, Iowa'),
    ],
    'KFK': [
        (1, 9999, 'US state law: Kansas, Kentucky'),
    ],
    'KFL': [
        (1, 999, 'US state law: Louisiana'),
    ],
    'KFM': [
        (1, 9999, 'US state law: Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, Missouri, Montana'),
    ],
    'KFN': [
        (1, 9999, 'US state law: Nebraska, Nevada, New Hampshire, New Jersey, New Mexico, New York, North Carolina, North Dakota'),
    ],
    'KFO': [
        (1, 999, 'US state law: Ohio, Oklahoma, Oregon'),
    ],
    'KFP': [
        (1, 999, 'US state law: Pennsylvania'),
    ],
    'KFR': [
        (1, 999, 'US state law: Rhode Island'),
    ],
    'KFS': [
        (1, 999, 'US state law: South Carolina, South Dakota'),
    ],
    'KFT': [
        (1, 999, 'US state law: Tennessee, Texas'),
    ],
    'KFU': [
        (1, 999, 'US state law: Utah'),
    ],
    'KFV': [
        (1, 9999, 'US state law: Vermont, Virginia'),
    ],
    'KFW': [
        (1, 9999, 'US state law: Washington, West Virginia, Wisconsin, Wyoming'),
    ],
    'KFX': [
        (1, 9999, 'US local law (cities, counties)'),
    ],
    'KG': [
        (1, 9999, 'Latin America. Mexico. Central America. West Indies'),
    ],
    'KH': [
        (1, 9999, 'South America'),
    ],
    'KJ': [
        (1, 9999, 'Europe (general)'),
    ],
    'KJC': [
        (1, 9999, 'European regional, comparative & uniform law'),
    ],
    'KJK': [
        (1, 4990, 'Albania, Andorra'),
    ],
    'KJV': [
        (1, 9999, 'France'),
    ],
    'KK': [
        (1, 9999, 'Germany'),
    ],
    'KKA': [
        (1, 9999, 'Germany: federal & state'),
    ],
    'KKE': [
        (1, 9999, 'Greece, Hungary'),
    ],
    'KL': [
        (1, 9999, 'Asia & Eurasia'),
    ],
    'KM': [
        (1, 9999, 'South Asia'),
    ],
    'KN': [
        (1, 9999, 'East Asia (China, Japan, Korea)'),
    ],
    'KP': [
        (1, 9999, 'Southeast Asia'),
    ],
    'KQ': [
        (1, 9999, 'Africa'),
    ],
    'KR': [
        (1, 9999, 'Africa (Eastern, Southern)'),
    ],
    'KS': [
        (1, 9999, 'Pacific area & Antarctica'),
    ],
    'KU': [
        (1, 9999, 'Australia'),
    ],
    'KV': [
        (1, 9999, 'Other Pacific & Oceania'),
    ],
    'KW': [
        (1, 9999, 'Pacific area legal systems'),
    ],
    'KZ': [
        (2, 6795, 'International law'),
        (199, 1450, 'History of international law'),
        (1234, 1236, 'Treaties on international law'),
        (3092, 3093, 'Sources of international law'),
        (3110, 3775, 'Subjects of international law (states, peoples)'),
        (4002, 5490, 'Specific subjects'),
        (6010, 6795, 'International criminal law, war, peace'),
    ],

    # ===== Class L — Education =====
    'LA': [
        (5, 2396, 'History of education'),
    ],
    'LB': [
        (5, 3640, 'Theory & practice of education'),
        (1025, 1050.75, 'Teaching (Principles & practice)'),
        (1050.9, 1091, 'Educational psychology'),
        (1101, 1139, 'Child study'),
        (1140, 1140.5, 'Preschool education'),
        (1141, 1489, 'Kindergarten'),
        (1501, 1602, 'Elementary education'),
        (1603, 1696.6, 'Secondary education'),
        (1705, 2286, 'Education & training of teachers'),
        (2300, 2430, 'Higher education'),
    ],
    'LC': [
        (8, 6691, 'Special aspects of education'),
        (1390, 5160.3, 'Education of special classes of persons'),
    ],
    'LD': [
        (13, 7501, 'Individual U.S. institutions'),
    ],

    # ===== Class P — Language & Literature =====
    'P': [
        (1, 1091, 'Philology & linguistics (general)'),
    ],
    'PA': [
        (1, 5665, 'Classical philology, Greek & Latin literature'),
    ],
    'PB': [
        (1, 3029, 'Modern European languages (general)'),
    ],
    'PC': [
        (1, 5498, 'Romance languages'),
    ],
    'PE': [
        (1, 3729, 'English language'),
    ],
    'PG': [
        (1, 9198, 'Slavic, Baltic, Albanian languages & literatures'),
    ],
    'PL': [
        (1, 8844, 'Languages & literatures of Eastern Asia, Africa, Oceania'),
    ],
    'PN': [
        (1, 6790, 'Literature (general)'),
        (1600, 3299, 'Drama'),
        (4001, 4321, 'Oratory, elocution'),
        (4699, 5650, 'Journalism, the periodical press'),
        (6010, 6790, 'Collections of general literature'),
    ],
    'PQ': [
        (1, 3999, 'French literature'),
        (4001, 5999, 'Italian literature'),
        (6001, 8929, 'Spanish literature'),
        (9000, 9999, 'Portuguese literature'),
    ],
    'PR': [
        (1, 78, 'English literature (general)'),
        (83, 888, 'History & criticism'),
        (1098, 1799, 'Collections'),
        (1803, 2165, 'Anglo-Saxon, Old & Middle English'),
        (2199, 2405, '15th–16th century'),
        (2411, 2999, 'Shakespeare & his contemporaries'),
        (3291, 3785, '17th–18th century'),
        (4000, 5990, '19th–20th century'),
        (6000, 6049, '1961–2000'),
        (6050, 6076, '21st century'),
        (8309, 9680, 'Commonwealth & former colonial literatures'),
    ],
    'PS': [
        (1, 3576, 'American literature'),
        (700, 893, 'Colonial period through 1830'),
        (991, 3390, '19th century'),
        (3500, 3549, '1900–1960'),
        (3550, 3576, '1961–2000'),
        (3600, 3626, '21st century'),
    ],
    'PT': [
        (1, 9999, 'German, Dutch, Scandinavian literatures'),
    ],
    'PZ': [
        (1, 90, 'Fiction & juvenile belles lettres'),
    ],

    # ===== Class Q — Science =====
    'QA': [
        (1, 939, 'Mathematics (general)'),
        (75, 76.95, 'Computer science & computing'),
        (101, 145, 'Elementary mathematics, arithmetic'),
        (150, 272.5, 'Algebra'),
        (273, 280, 'Probability theory'),
        (276, 280, 'Statistics'),
        (299, 433, 'Analysis & calculus'),
        (440, 699, 'Geometry, topology'),
        (801, 939, 'Analytic mechanics'),
    ],
    'QB': [
        (1, 991, 'Astronomy'),
    ],
    'QC': [
        (1, 999, 'Physics'),
        (170, 197, 'Atomic physics, quantum'),
        (350, 467, 'Optics, light'),
        (501, 766, 'Electricity & magnetism'),
        (770, 798, 'Nuclear & particle physics'),
        (851, 999, 'Geophysics, meteorology'),
    ],
    'QD': [
        (1, 999, 'Chemistry'),
        (146, 197, 'Inorganic chemistry'),
        (241, 441, 'Organic chemistry'),
        (450, 731, 'Physical & theoretical chemistry'),
    ],
    'QE': [
        (1, 996.5, 'Geology'),
    ],
    'QH': [
        (1, 705.5, 'Natural history (general)'),
        (301, 705.5, 'Biology'),
        (351, 425, 'General biology'),
        (426, 470, 'Genetics'),
        (471, 489, 'Reproduction'),
        (501, 531, 'Life'),
        (540, 599.9, 'Ecology'),
        (705, 705.5, 'Microscopy'),
    ],
    'QK': [
        (1, 989, 'Botany'),
    ],
    'QL': [
        (1, 991, 'Zoology'),
    ],
    'QM': [
        (1, 695, 'Human anatomy'),
    ],
    'QP': [
        (1, 981, 'Physiology'),
    ],
    'QR': [
        (1, 502, 'Microbiology'),
    ],

    # ===== Class R — Medicine =====
    'R': [
        (1, 920, 'Medicine (general)'),
    ],
    'RA': [
        (1, 1270, 'Public aspects of medicine'),
        (407, 409.5, 'Health status indicators, surveys'),
        (421, 790.95, 'Public health, hygiene, preventive medicine'),
        (1001, 1270, 'Forensic medicine, medical jurisprudence'),
    ],
    'RB': [
        (1, 214, 'Pathology'),
    ],
    'RC': [
        (1, 1245, 'Internal medicine, clinical medicine'),
        (31, 1245, 'Internal medicine'),
        (321, 571, 'Neurology, neurosciences'),
        (435, 571, 'Psychiatry'),
        (581, 951, 'Specialties (cardiovascular, oncology, etc.)'),
        (952, 1245, 'Geriatrics, sports medicine, tropical medicine'),
    ],
    'RD': [
        (1, 811, 'Surgery'),
    ],
    'RE': [
        (1, 994, 'Ophthalmology'),
    ],
    'RF': [
        (1, 547, 'Otorhinolaryngology'),
    ],
    'RG': [
        (1, 991, 'Gynecology & obstetrics'),
    ],
    'RJ': [
        (1, 570, 'Pediatrics'),
    ],
    'RT': [
        (1, 120, 'Nursing'),
    ],

    # ===== Class T — Technology =====
    'T': [
        (1, 995, 'Technology (general)'),
        (10.5, 11.9, 'Communication of technical information'),
        (55, 55.3, 'Industrial safety, hygiene'),
        (170, 174.5, 'Industrial archaeology'),
        (385, 388, 'Engineering & technical drawing'),
    ],
    'TA': [
        (1, 2040, 'Engineering (general), civil engineering'),
        (1, 145, 'General engineering'),
        (164, 167, 'Bioengineering'),
        (168, 168, 'Systems engineering'),
        (170, 171, 'Environmental engineering'),
        (174, 174, 'Engineering design'),
        (177.4, 185, 'Engineering economy'),
        (190, 197, 'Engineering operations & management'),
        (213, 215, 'Engineering machinery & tools'),
        (329, 348, 'Engineering mathematics'),
        (349, 359, 'Mechanics of engineering'),
        (401, 492, 'Materials of engineering & construction'),
        (501, 625, 'Surveying'),
        (630, 695, 'Structural engineering'),
        (703, 712, 'Engineering geology'),
        (715, 787, 'Earthwork. Foundations'),
        (800, 820, 'Tunneling'),
        (1001, 1280, 'Transportation engineering'),
        (1501, 1820, 'Applied optics. Photonics'),
        (2001, 2040, 'Plasma engineering'),
    ],
    'TC': [
        (1, 1800, 'Hydraulic engineering'),
        (1501, 1800, 'Ocean engineering'),
    ],
    'TD': [
        (1, 1066, 'Environmental technology, sanitary engineering'),
        (159, 168, 'Municipal engineering'),
        (169, 171.8, 'Environmental protection'),
        (172, 193.5, 'Environmental pollution'),
        (194, 195, 'Environmental impact analysis'),
        (201, 500, 'Water supply for domestic & industrial purposes'),
        (511, 780, 'Sewage collection & disposal'),
        (783, 812.5, 'Municipal refuse, solid wastes'),
        (878, 894, 'Special types of environment'),
        (895, 899, 'Industrial sanitation, industrial hygiene'),
        (920, 934, 'Rural & farm sanitary engineering'),
        (940, 949, 'Low temperature sanitary engineering'),
    ],
    'TE': [
        (1, 450, 'Highway engineering, roads & pavements'),
    ],
    'TF': [
        (1, 1620, 'Railroad engineering & operation'),
    ],
    'TG': [
        (1, 470, 'Bridge engineering'),
    ],
    'TH': [
        (1, 9745, 'Building construction'),
        (5011, 5701, 'Construction by phase of work'),
        (7005, 7699, 'Heating & ventilation. Air conditioning'),
        (7700, 7975, 'Illumination. Lighting'),
        (8001, 8581, 'Decoration & decorative furnishings'),
        (9025, 9745, 'Protection of buildings'),
    ],
    'TJ': [
        (1, 1570, 'Mechanical engineering & machinery'),
        (170, 179, 'Mechanics applied to machinery. Dynamics'),
        (181, 210, 'Mechanical movements'),
        (212, 225, 'Mechanical drives'),
        (227, 240, 'Machine design & drawing'),
        (241, 254.7, 'Machine construction'),
        (255, 265, 'Heat engines'),
        (266, 267.5, 'Turbines. Turbomachines'),
        (268, 740, 'Steam engineering'),
        (751, 805, 'Locomotives'),
        (807, 830, 'Power resources'),
        (836, 927, 'Hydraulic machinery'),
        (940, 940, 'Vacuum technology'),
        (950, 1030, 'Pneumatic machinery'),
        (1040, 1119, 'Machinery exclusive of prime movers'),
        (1125, 1345, 'Machine shops & machine shop practice'),
        (1380, 1495, 'Hoisting & conveying machinery'),
        (1517, 1519, 'Lubrication & lubricants'),
        (1525, 1570, 'Mechatronics. Microelectromechanical systems. Robots'),
    ],
    'TK': [
        (1, 9971, 'Electrical engineering, electronics, nuclear engineering'),
        (5101, 6720.5, 'Telecommunications'),
        (7800, 8360, 'Electronics'),
        (7885, 7895, 'Computer engineering'),
        (9001, 9401, 'Nuclear engineering. Atomic power'),
    ],
    'TL': [
        (1, 4050, 'Motor vehicles, aeronautics, astronautics'),
        (1, 484, 'Motor vehicles. Cycles'),
        (500, 777, 'Aeronautics. Aeronautical engineering'),
        (780, 785.8, 'Rocket propulsion. Rockets'),
        (787, 4050, 'Astronautics. Space travel'),
    ],
    'TN': [
        (1, 997, 'Mining engineering. Metallurgy'),
    ],
    'TP': [
        (1, 1185, 'Chemical technology'),
        (155, 156, 'Chemical engineering'),
        (368, 456, 'Food processing & manufacture'),
        (480, 498, 'Low temperature engineering. Cryogenic engineering'),
        (500, 660, 'Fermentation industries. Beverages. Alcohol'),
        (670, 699, 'Oils, fats, waxes'),
        (700, 746, 'Illuminating industries (non-electric)'),
        (785, 869, 'Clay industries. Ceramics. Glass'),
        (875, 888, 'Cement industries'),
        (890, 933, 'Explosives & pyrotechnics'),
        (934, 945, 'Paints, pigments, varnishes'),
        (1080, 1185, 'Polymers & polymer manufacture'),
    ],
    'TR': [
        (1, 1050, 'Photography'),
        (1, 196, 'Photography (general)'),
        (200, 559, 'Photographic processing & materials'),
        (624, 835, 'Applied photography'),
        (845, 899, 'Cinematography. Motion pictures'),
        (925, 1050, 'Photomechanical processes'),
    ],
    'TS': [
        (1, 2301, 'Manufactures'),
        (155, 194, 'Production management'),
        (200, 770, 'Metal manufactures. Metalworking'),
        (800, 937, 'Wood technology. Lumber'),
        (940, 1047, 'Leather industries. Tanning'),
        (1060, 1070, 'Furs'),
        (1080, 1268, 'Paper manufacture & trade'),
        (1300, 1865, 'Textile industries'),
        (1870, 1935, 'Rubber industry'),
        (1950, 2301, 'Animal products'),
    ],
    'TT': [
        (1, 999, 'Handicrafts. Arts & crafts'),
        (161, 170.7, 'Manual training. School shops'),
        (174, 176, 'Articles for children'),
        (180, 200, 'Woodworking. Furniture making'),
        (201, 203, 'Lathework. Turning'),
        (205, 267, 'Metalworking'),
        (300, 387, 'Painting. Wood finishing'),
        (387, 410, 'Soft home furnishings'),
        (490, 695, 'Clothing manufacture. Dressmaking. Tailoring'),
        (697, 927, 'Home arts. Homecrafts'),
        (950, 979, 'Hairdressing. Beauty culture. Barbers\' work'),
        (980, 999, 'Laundry work'),
    ],
    'TX': [
        (1, 1110, 'Home economics'),
        (301, 339, 'The house'),
        (341, 641, 'Nutrition. Foods & food supply'),
        (642, 840, 'Cooking'),
        (851, 885, 'Dining-room service'),
        (901, 953, 'Hospitality industry. Hotels, clubs, restaurants'),
        (955, 985, 'Mobile home living. Trailer camps. Recreational vehicles'),
    ],

    # ===== Class S — Agriculture =====
    'S': [
        (1, 972, 'Agriculture (general)'),
        (560, 575.5, 'Farm economics. Farm management'),
        (583, 587.73, 'Agricultural chemistry'),
        (605.5, 605.5, 'Organic farming'),
        (621, 621.5, 'Agricultural meteorology'),
        (622, 627, 'Soils. Soil science'),
        (631, 667, 'Fertilizers'),
        (671, 760.5, 'Farm machinery & engineering'),
        (900, 972, 'Conservation of natural resources'),
    ],
    'SB': [
        (1, 1110, 'Plant culture'),
        (109, 110, 'Economic botany'),
        (175, 423, 'Field crops'),
        (450, 467.8, 'Horticulture. Horticultural crops'),
        (469, 476.4, 'Landscape architecture'),
        (599, 990.5, 'Pests & diseases'),
        (599, 990.5, 'Pests & diseases'),
        (992, 998, 'Economic entomology'),
    ],
    'SD': [
        (1, 669.5, 'Forestry'),
    ],
    'SF': [
        (1, 1100, 'Animal culture'),
        (411, 459, 'Pets'),
        (481, 507, 'Poultry. Eggs'),
        (600, 1100, 'Veterinary medicine'),
    ],
    'SH': [
        (1, 691, 'Aquaculture. Fisheries. Angling'),
    ],
    'SK': [
        (1, 664, 'Hunting sports'),
    ],

    # ===== Class U — Military Science =====
    'U': [
        (1, 900, 'Military science (general)'),
        (21, 22, 'Theory of military science'),
        (101, 145, 'History of military science'),
        (159, 165, 'War. Philosophy. Military sociology'),
        (200, 305, 'Strategy & tactics'),
        (400, 714, 'Military education & training'),
        (750, 773, 'Military life, customs, morale'),
        (799, 897, 'Military administration'),
    ],
    'UA': [
        (10, 997, 'Armies: organization, distribution, military situation'),
    ],
    'UB': [
        (1, 900, 'Military administration'),
        (160, 165, 'Records, returns, muster rolls'),
        (170, 175, 'Adjutant generals\' offices'),
        (180, 197, 'Inspection'),
        (250, 271, 'Intelligence'),
        (275, 277, 'Espionage'),
        (320, 615, 'Personnel management'),
        (407, 409, 'Compulsory service. Conscription'),
        (416, 419, 'Voluntary service'),
        (663, 665, 'Veterans'),
    ],
    'UC': [
        (10, 780, 'Maintenance & transportation'),
    ],
    'UD': [
        (1, 495, 'Infantry'),
    ],
    'UE': [
        (1, 500, 'Cavalry. Armor'),
    ],
    'UF': [
        (1, 910, 'Artillery'),
    ],
    'UG': [
        (1, 5000, 'Military engineering. Air forces. Space surveillance'),
        (570, 614, 'Air forces. Air warfare'),
        (622, 1435, 'Military air forces by region or country'),
        (1500, 1530, 'Air defenses'),
        (1900, 1950, 'Military space surveillance & operations'),
    ],
    'UH': [
        (20, 800, 'Other services'),
    ],

    # ===== Class V — Naval Science =====
    'V': [
        (1, 995, 'Naval science (general)'),
    ],
    'VA': [
        (10, 750, 'Navies: organization, description, facilities'),
    ],
    'VB': [
        (15, 970, 'Naval administration'),
    ],
    'VC': [
        (10, 580, 'Naval maintenance'),
    ],
    'VD': [
        (7, 430, 'Naval seamen'),
    ],
    'VE': [
        (7, 500, 'Marines'),
    ],
    'VF': [
        (1, 580, 'Naval ordnance'),
    ],
    'VG': [
        (20, 2029, 'Minor services of navies'),
    ],
    'VK': [
        (1, 1661, 'Navigation. Merchant marine'),
    ],
    'VM': [
        (1, 989, 'Naval architecture. Shipbuilding. Marine engineering'),
    ],

    # ===== Class M — Music =====
    'ML': [
        (1, 3930, 'Literature on music'),
    ],
    'M': [
        (1, 5000, 'Music'),
    ],

    # ===== Class N — Fine Arts =====
    'N': [
        (1, 9165, 'Visual arts (general)'),
    ],
    'NA': [
        (1, 9428, 'Architecture'),
    ],
    'NB': [
        (1, 1952, 'Sculpture'),
    ],
    'NC': [
        (1, 1940, 'Drawing, design, illustration'),
    ],
    'ND': [
        (25, 3416, 'Painting'),
    ],
    'NE': [
        (1, 3002, 'Print media'),
    ],

    # ===== Class Z — Bibliography & Library Science =====
    'Z': [
        (4, 1, 'Books in general'),
        (40, 115.5, 'Writing, paleography'),
        (116, 659, 'Book industries & trade'),
        (665, 1000.5, 'Libraries & library science'),
        (1001, 8999, 'Bibliography'),
    ],
}


def _bucket_by_hundreds(num):
    """Generic fallback for subclasses without a curated range catalog.

    Buckets a numeric LC subclass component into '0s', '100s', '200s', etc.
    Used when LC_RANGES has no entry for the relevant subclass.
    """
    if num is None or pd.isna(num):
        return None
    try:
        hundred = int(num // 100) * 100
        return f"{hundred}s" if hundred > 0 else "0s"
    except (ValueError, TypeError):
        return None


def lookup_lc_range(subclass, number):
    """Map a (subclass, number) pair to its range label.

    Walks the LC_RANGES catalog for the subclass; if the number falls inside
    a curated range, returns that range's label. If no curated range exists
    or no range matches, falls back to hundreds-bucketing (e.g., 'HQ 1100s').
    Returns None when subclass or number is missing.

    Note: ranges within a subclass may overlap (e.g., HD 4801-8943 and
    HD 5001-5840 both exist in the official LC outline). When that happens,
    the *more specific* (narrower) range wins, so callers get the most
    informative label available.
    """
    if not subclass or number is None or pd.isna(number):
        return None
    ranges = LC_RANGES.get(subclass.upper())
    if not ranges:
        bucket = _bucket_by_hundreds(number)
        return f"{subclass} {bucket}" if bucket else None

    # Find all matching ranges, then prefer the narrowest (most specific)
    matches = [(start, end, label) for (start, end, label) in ranges
               if start <= number <= end]
    if matches:
        # Sort by width ascending — narrower range wins
        matches.sort(key=lambda r: r[1] - r[0])
        return matches[0][2]

    # No curated range matched; fall back to hundreds-bucketing
    bucket = _bucket_by_hundreds(number)
    return f"{subclass} {bucket}" if bucket else None



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
    """Vectorized LC call number parsing.

    Returns three Series — (main_class, subclass, number) — extracted from
    call numbers like 'HQ1190 .C66 2007':
      - main_class: first letter ('H')
      - subclass:   1–3 leading letters ('HQ')
      - number:     numeric component as float ('1190.0'); supports decimals
                    like 'HQ1090.7' which appear in some LC ranges. None when
                    the call number has no parseable numeric component.

    The numeric component enables sub-class range analysis via lookup_lc_range.
    """
    cleaned = series.astype(str).str.strip().str.upper()
    # Combined regex: capture letters + optional decimal number
    extracted = cleaned.str.extract(r'^([A-Z]{1,3})\s*([0-9]+(?:\.[0-9]+)?)?', expand=True)
    letters = extracted[0]
    number_str = extracted[1]
    main_class = letters.str[0]
    # Convert number to float; non-matches become NaN
    number = pd.to_numeric(number_str, errors='coerce')
    mask = series.isna() | (series.astype(str).str.strip() == '')
    main_class = main_class.where(~mask, other=None)
    letters = letters.where(~mask, other=None)
    number = number.where(~mask, other=None)
    return main_class, letters, number


# Title-keyword analysis ------------------------------------------------------
# A separate lens from subject headings: titles are uncontrolled vocabulary, so
# we strip a generous stopword list (English + publishing/library noise) before
# counting. This view is supplementary — it surfaces vocabulary that subject
# headings missed (newer concepts, methodological terms, interdisciplinary
# phrases) without diluting the controlled-vocabulary subject analysis.
TITLE_STOPWORDS = frozenset({
    # Articles, conjunctions, prepositions, pronouns, common verbs
    'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet', 'for',
    'of', 'in', 'on', 'at', 'to', 'from', 'by', 'with', 'without', 'about',
    'as', 'into', 'onto', 'upon', 'over', 'under', 'through', 'across',
    'after', 'before', 'between', 'among', 'against', 'during', 'until',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'has', 'have', 'had', 'having', 'do', 'does', 'did', 'doing',
    'this', 'that', 'these', 'those', 'it', 'its', "it's",
    'i', 'you', 'he', 'she', 'we', 'they', 'them', 'us', 'our', 'their',
    'his', 'her', 'my', 'your',
    'not', 'no', 'yes', 'if', 'then', 'than', 'when', 'where', 'why', 'how',
    'what', 'which', 'who', 'whom', 'whose',
    'all', 'any', 'some', 'each', 'every', 'other', 'another', 'such',
    'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must',
    'shall', 'one', 'two', 'three', 'first', 'second', 'third',
    # Generic publishing / book-title noise
    'introduction', 'introductory', 'guide', 'handbook', 'companion',
    'reader', 'manual', 'primer', 'textbook', 'workbook', 'casebook',
    'overview', 'survey', 'review', 'reviews', 'essays', 'essay',
    'studies', 'study', 'research', 'researches', 'analysis', 'analyses',
    'approach', 'approaches', 'perspective', 'perspectives',
    'edition', 'ed', 'eds', 'editor', 'editors', 'edited', 'rev', 'revised',
    'volume', 'vol', 'vols', 'series', 'collection', 'selected', 'works',
    'new', 'newer', 'newest', 'modern', 'contemporary', 'recent',
    'practical', 'theoretical', 'theory', 'theories',
    'principles', 'principle', 'fundamentals', 'fundamental', 'basics', 'basic',
    'concepts', 'concept', 'topics', 'topic', 'issues', 'issue',
    'history', 'introduction', 'making', 'understanding',
    # Single letters (strays from initials, "u s" from "U.S.", roman numerals)
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix', 'xi', 'xii',
    # Subtitle / format words common in book titles
    'using', 'use', 'used', 'within', 'across', 'toward', 'towards',
    'more', 'most', 'less', 'least', 'much', 'many', 'few', 'fewer',
    'years', 'year', 'century', 'centuries', 'decade', 'decades',
    'today', 'tomorrow', 'yesterday',
    'also', 'just', 'only', 'even', 'still', 'already', 'always', 'never',
    'really', 'actually', 'simply', 'indeed', 'thus', 'hence',
    'beyond', 'around', 'something', 'someone', 'anything', 'nothing',
})

_RE_TITLE_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+")
# Split a title into segments at subtitle/clause boundaries so n-grams don't
# cross conceptual breaks (e.g., "Insurgent Cuba: Race, Nation, and Revolution"
# should not produce the bigram "cuba race"). Splits on : ; — – and commas.
_RE_TITLE_SEGMENT = re.compile(r"[:;,()\[\]\u2013\u2014/\\]+|\s-\s")


def _tokenize_title_segments(title, min_len=4, extra_stopwords=None):
    """Return a list of token-lists, one per subtitle/clause segment.

    N-gram extraction respects segment boundaries, so phrases like 'race nation'
    won't be formed across a colon or comma. Within each segment, stopwords
    and short tokens are dropped.
    """
    if not isinstance(title, str) or not title.strip():
        return []
    stops = TITLE_STOPWORDS if not extra_stopwords else (TITLE_STOPWORDS | extra_stopwords)
    segments_raw = _RE_TITLE_SEGMENT.split(title.lower())
    segments = []
    for seg in segments_raw:
        raw_tokens = _RE_TITLE_TOKEN.findall(seg)
        kept = []
        for tok in raw_tokens:
            tok = tok.strip("'-")
            if len(tok) < min_len:
                continue
            if tok in stops:
                continue
            if tok.isdigit():
                continue
            kept.append(tok)
        if kept:
            segments.append(kept)
    return segments


def _tokenize_title(title, min_len=4, extra_stopwords=None):
    """Flat token list (preserves segment-aware filtering). Kept for callers
    that just want unigrams without segment structure."""
    return [tok for seg in _tokenize_title_segments(title, min_len, extra_stopwords)
            for tok in seg]


def _extract_ngrams(segments, n_values=(1,)):
    """Yield (n, ngram_string) tuples from segment-aware token lists.

    n_values controls which sizes to extract. N-grams of size n require a
    segment of length ≥ n; they never cross segment boundaries.
    """
    for seg in segments:
        for n in n_values:
            if n < 1 or len(seg) < n:
                continue
            for i in range(len(seg) - n + 1):
                yield n, ' '.join(seg[i:i + n])


# =====================================================================
# SHARED: Column detection & CSV loading
# =====================================================================

SUBJECT_ALIASES = ['Subjects', 'Subject', 'Subject Terms', 'Subject Headings',
                   'SUBJECT', 'subject_terms', 'Topics', 'subjects', 'Topic',
                   'LCSH', 'Library of Congress Subject', 'Subject (LCSH)']
LC_ALIASES = ['LC Classification Code', 'LC Classification', 'LC Class',
              'LC Subclass', 'LCC',
              'Call Number', 'CallNumber', 'call_number', 'Call #',
              'Permanent Call Number', 'Normalized Call Number',
              'LC Call Number', 'Classification', 'lc_classification',
              'Library of Congress Classification']
TITLE_ALIASES = ['Title', 'title', 'TITLE', 'Book Title', 'Item Title',
                 'Title (Normalized)', 'File Name', 'Filename',
                 'Item Name', 'Resource Title', 'Object Title']

# Weight/usage aliases — ORDER MATTERS. More specific and meaningful usage
# columns appear first so they win the alias contest before generic ones.
# 'Uses' (bare) deliberately omitted — too generic; matches "Remaining CAM Uses"
# which is *available* capacity, not actual usage.
WEIGHT_ALIASES = [
    # COUNTER 5 metrics (most specific — formal e-resource usage)
    'Total_Item_Requests', 'Unique_Item_Requests',
    'Total_Item_Investigations', 'Unique_Item_Investigations',
    'Total_Title_Requests', 'Unique_Title_Requests',
    # EBSCO Detailed Report — actual usage (in preference order)
    'Total Accesses', 'Full Downloads', 'Chapter Downloads', 'Online Views',
    # Print circulation
    'Loans (Total)', 'Loans (In House + Not In House)',
    'Loans', 'Checkouts', 'Circulation', 'checkouts',
    # Digital views / downloads
    'Digital File Downloads', 'Digital File Views',
    'Total Book Downloads', 'Book Downloads', 'Downloads',
    'Read Online (post Trigger) Sessions',
    'Read Online Sessions', 'Sessions',
    # Generic — last resort
    'Views', 'Requests', 'Hits', 'Usage', 'Total Uses', 'Count',
]

# Identifier columns for cross-file matching (used by Zero-Use Identifier).
# ISBN/ISSN/DOI/OCLC are reliable join keys; title+author is the fallback
# when identifiers are absent.
ISBN_ALIASES = ['ISBN', 'isbn', 'ISBN-13', 'ISBN13', 'ISBN-10', 'ISBN10',
                'eISBN', 'Print ISBN', 'Online ISBN', 'Print_ISBN', 'Online_ISBN']
ISSN_ALIASES = ['ISSN', 'issn', 'eISSN', 'Print ISSN', 'Online ISSN',
                'Print_ISSN', 'Online_ISSN']
DOI_ALIASES = ['DOI', 'doi', 'DOI Link']
OCLC_ALIASES = ['OCLC', 'OCLC Number', 'OCLC #', 'OCLC_Number',
                'WorldCat Number', 'OCN']
AUTHOR_ALIASES = ['Author', 'author', 'AUTHOR', 'Creator', 'Authors',
                  'Primary Author', 'Main Author']
LOCATION_ALIASES = ['Location', 'Location Name', 'location', 'Library Location',
                    'Shelving Location', 'Holding Location']

# Used by the Overlap & Uniqueness tool (e-journal coverage / A-Z exports).
COVERAGE_ALIASES = ['Coverage Information Combined', 'Coverage Information',
                    'Coverage Statement', 'Coverage Combined', 'Available Coverage',
                    'Date Coverage', 'Coverage Dates', 'Coverage']
COLLECTION_ALIASES = ['Electronic Collection Public Name', 'Electronic Collection',
                      'Collection Public Name', 'Public Name', 'Collection Name',
                      'Package Name', 'Database Name', 'Resource Name',
                      'Collection', 'Package', 'Database']
INTERFACE_ALIASES = ['Interface Name', 'Interface', 'Provider Name', 'Provider',
                     'Platform', 'Vendor', 'Service Provider']
NORM_TITLE_ALIASES = ['Title (Normalized)', 'Normalized Title', 'Title Normalized',
                      'Title (normalized)']


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


def _count_leading_comment_lines(file_bytes, comment='#'):
    """Count contiguous leading '#'-comment lines (a provenance header).

    Our exports — e.g. the Zero-Use explicit-zero master — prepend a '#' metadata
    block via _annotate_csv. Counting only *leading* comment lines lets the loaders
    skip that block when the file is re-ingested, without disturbing a '#' that
    appears inside a real data field. The blank separator after the block is
    handled by pandas' skip_blank_lines.
    """
    head = file_bytes[:65536]
    try:
        text = head.decode('utf-8-sig', errors='replace')
    except Exception:
        text = head.decode('latin-1', errors='replace')
    n = 0
    for line in text.splitlines():
        if line.lstrip('\ufeff').lstrip().startswith(comment):
            n += 1
        else:
            break
    return n


def _detect_columns_from_header(file_bytes, filename=None):
    """Read only the header row to detect column names without loading all data.

    Dispatches on filename extension when provided — CSV uses read_csv, XLS/XLSX
    uses read_excel. Falls back to CSV if filename is missing.
    """
    if filename and filename.lower().endswith(('.xls', '.xlsx')):
        try:
            header = pd.read_excel(BytesIO(file_bytes), nrows=0)
            return [c.strip() if isinstance(c, str) else c for c in header.columns]
        except Exception:
            pass  # fall through to CSV attempt
    skip = _count_leading_comment_lines(file_bytes)
    try:
        header = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', nrows=0, skiprows=skip)
    except Exception:
        header = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', nrows=0, skiprows=skip)
    return [c.strip() for c in header.columns]


@st.cache_data(show_spinner=False)
def _load_csv_chunked(file_bytes, filename, cols_to_keep=None):
    """Load CSV or Excel file with minimal memory footprint.

    Despite the name (kept for backward compatibility with cache keys), this now
    dispatches based on filename extension: .xls/.xlsx use pandas.read_excel,
    everything else uses read_csv with utf-8-sig → latin-1 fallback.
    """
    if filename and filename.lower().endswith(('.xls', '.xlsx')):
        # Excel path — usecols works the same way as CSV
        try:
            df = pd.read_excel(BytesIO(file_bytes), usecols=cols_to_keep)
        except Exception:
            # If cols_to_keep failed (e.g., mismatch), try without it
            df = pd.read_excel(BytesIO(file_bytes))
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return df

    # CSV path (original behavior). Our own exports (e.g. the Zero-Use
    # explicit-zero master) carry a leading '#'-comment provenance block from
    # _annotate_csv; skip those lines so the file round-trips cleanly. Only
    # *leading* comment lines are skipped, so a '#' inside a data field survives.
    skip = _count_leading_comment_lines(file_bytes)
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False,
                         skiprows=skip, usecols=cols_to_keep)
    except Exception:
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', low_memory=False,
                             skiprows=skip, usecols=cols_to_keep)
        except Exception:
            try:
                df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False,
                                 skiprows=skip)
            except Exception:
                df = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', low_memory=False,
                                 skiprows=skip)
    df.columns = df.columns.str.strip()
    return df


# =====================================================================
# SHARED: Footer & decision-box helper
# =====================================================================

def _footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Library Collection Dashboard v2.0  built with Claude| Howard-Tilton Memorial Library, Tulane University</p>
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
# SHARED: Session caching (across tool switches)
# =====================================================================
# Streamlit's @st.cache_data handles the raw CSV parse, but we also want to
# cache the post-processing work (LC extraction, weight coercion, etc.) so
# switching between tools and coming back doesn't force a re-setup.
# Key scheme: one cache slot per tool, keyed by (filename, filesize).

def _make_file_key(uploaded_file):
    """Build a stable cache key from an uploaded file object."""
    if uploaded_file is None:
        return None
    try:
        return (uploaded_file.name, uploaded_file.size)
    except AttributeError:
        # Fallback for file-like objects without .size
        return (uploaded_file.name, None)


def _cached_df_for_tool(tool_key, uploaded_file):
    """Retrieve a cached processed DataFrame for this tool+file, if it exists.

    Returns the cached df, or None if nothing matches (caller should do the load).
    """
    cache_key = f"_df_cache_{tool_key}"
    file_key = _make_file_key(uploaded_file)
    cached = st.session_state.get(cache_key)
    if cached and cached.get('file_key') == file_key:
        return cached.get('df')
    return None


def _store_cached_df(tool_key, uploaded_file, df):
    """Store a processed DataFrame in session state for this tool+file."""
    cache_key = f"_df_cache_{tool_key}"
    st.session_state[cache_key] = {
        'file_key': _make_file_key(uploaded_file),
        'df': df,
    }


# =====================================================================
# SHARED: Analysis notes
# =====================================================================
# Users can attach free-text notes to any analysis; these travel with downloads
# as a header comment and can be reviewed later. Notes persist per tool in
# session state, keyed by tool so switching tools doesn't lose them.

def _notes_widget(tool_key, label="📝 Analysis notes", placeholder=None):
    """Render a notes text area and return its current value.

    The value persists in session_state so it survives reruns and tool switches.
    Intended to be called near the top of each tool's analysis output so users
    can annotate *before* downloading.
    """
    note_key = f"_notes_{tool_key}"
    if note_key not in st.session_state:
        st.session_state[note_key] = ""

    placeholder = placeholder or (
        "e.g., Prepared for sociology liaison meeting, Nov 2025. "
        "Follow-up: discuss HQ underperformance with Dr. Chen."
    )

    with st.expander(label, expanded=False):
        st.caption("Notes are saved in this session and included as a header comment "
                   "in any CSV you download below. They won't persist if you close "
                   "the browser tab.")
        notes = st.text_area(
            "Add context, rationale, or follow-up items:",
            value=st.session_state[note_key],
            placeholder=placeholder,
            key=f"{note_key}_widget",
            height=100,
        )
        st.session_state[note_key] = notes
    return notes


def _annotate_csv(df, notes, extra_meta=None):
    """Return CSV bytes with an optional notes header block prepended.

    The notes appear as CSV comment lines (prefixed with #) which Excel reads
    as a single first row but most CSV libraries skip. Kept simple and portable.
    """
    from io import StringIO
    from datetime import datetime

    lines = []
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"# {k}: {v}")
    if notes and notes.strip():
        lines.append("# Notes:")
        for ln in notes.strip().splitlines():
            lines.append(f"#   {ln}")
    lines.append("")  # blank line before CSV body

    buf = StringIO()
    if lines:
        buf.write("\n".join(lines) + "\n")
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _zip_one_csv(csv_data, inner_filename):
    """Wrap a single CSV (str or bytes) into deflated ZIP bytes.

    Compressing the download shrinks a large title-level export to a small
    fraction of its raw size, keeping transfers reliable on memory-capped
    hosts. A single-file .zip stays friendly for Excel users on Windows.
    """
    import zipfile
    from io import BytesIO as _BIO
    if isinstance(csv_data, str):
        csv_data = csv_data.encode("utf-8")
    buf = _BIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_filename, csv_data)
    return buf.getvalue()


# =====================================================================
# SHARED: Download tray
# =====================================================================
# Collects every downloadable artifact produced during a single tool run
# so users can grab everything as one ZIP at the end of the page instead
# of hunting for individual buttons.
#
# Usage: alongside an existing st.download_button, call
#     _add_to_tray(tool_key, filename, data)
# Then at the end of the tool's output, call
#     _render_download_tray(tool_key)
# to render a "Download all (ZIP)" button.

def _reset_tray(tool_key):
    """Clear the tray for this tool. Call at the start of a fresh render pass
    so stale artifacts from a previous run don't leak into the ZIP."""
    st.session_state[f"_tray_{tool_key}"] = []


def _add_to_tray(tool_key, filename, data):
    """Register a downloadable artifact (CSV string or bytes) for this tool."""
    tray_key = f"_tray_{tool_key}"
    if tray_key not in st.session_state:
        st.session_state[tray_key] = []
    # De-dup: if this filename is already in the tray, overwrite it
    tray = st.session_state[tray_key]
    tray[:] = [item for item in tray if item[0] != filename]
    tray.append((filename, data))


def _render_download_tray(tool_key, zip_filename="results.zip"):
    """Render a 'Download all' button that bundles everything in the tray."""
    tray = st.session_state.get(f"_tray_{tool_key}", [])
    if not tray:
        return
    import zipfile
    from io import BytesIO as _BIO
    buf = _BIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in tray:
            if isinstance(data, str):
                data = data.encode("utf-8")
            zf.writestr(filename, data)
    buf.seek(0)
    count = len(tray)
    st.download_button(
        f"📦 Download all ({count} file{'s' if count != 1 else ''}) as ZIP",
        buf.getvalue(),
        zip_filename,
        "application/zip",
        key=f"_tray_dl_{tool_key}",
        use_container_width=True,
        type="primary",
    )
    with st.expander(f"Files included ({count})", expanded=False):
        for filename, _ in tray:
            st.caption(f"• {filename}")


def _dl(label, data, filename, mime, key, tool_key=None):
    """Render a download button AND stash it in the tool's tray.

    Thin wrapper over st.download_button so existing call sites can be
    converted with minimal change.
    """
    st.download_button(label, data, filename, mime, key=key)
    if tool_key is not None:
        _add_to_tray(tool_key, filename, data)


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
                           selected_classes, progress_bar,
                           has_usage_col=False, ngram_sizes=(1, 2, 3),
                           lc_filter_active=False):
    """Single pass that builds everything: LC counts, subject counter, subject-by-LC, gaps.

    When a usage column is present in the dataframe (indicated by `has_usage_col=True`),
    also computes a second set of LC counts using usage weighting — this is what powers
    the Coverage-vs-Use view. The weight_col parameter controls the primary analysis;
    the secondary pass always uses the '_weight' column (which holds usage values).

    `ngram_sizes` controls which title-keyword n-gram sizes are extracted
    (1=words, 2=bigrams, 3=trigrams). All sizes are stored together in a single
    counter keyed by (size, phrase); the renderer filters by size.
    """
    n_total = len(df)
    if selected_classes is not None and lc_col:
        mask = df['_lc_main'].isin(selected_classes)
        # When the filter isn't actively narrowed (all classes selected), also let
        # unclassified records through so the default view covers everything. When
        # the user HAS narrowed it, exclude unclassified so every view — including
        # the subject word cloud — strictly reflects the chosen classes.
        if not lc_filter_active:
            mask = mask | df['_lc_main'].isna()
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

        # Coverage vs. Use: compute BOTH title-count and usage-weighted distributions
        # (regardless of primary weight_col) when a usage column is present
        if has_usage_col:
            # Title counts (always weight=1)
            titles_main = pd.Series(1.0, index=lc_main.index).groupby(lc_main).sum().to_dict()
            titles_sub = pd.Series(1.0, index=lc_sub.index).groupby(lc_sub).sum().to_dict()
            # Usage counts (always from _weight column, which holds the actual usage values)
            usage_series = df.loc[idx, '_weight']
            usage_main_w = usage_series.reindex(lc_main.index)
            usage_sub_w = usage_series.reindex(lc_sub.index)
            usage_main = usage_main_w.groupby(lc_main).sum().to_dict()
            usage_sub = usage_sub_w.groupby(lc_sub).sum().to_dict()
            results['cvu_available'] = True
            results['cvu_titles_main'] = titles_main
            results['cvu_titles_sub'] = titles_sub
            results['cvu_usage_main'] = usage_main
            results['cvu_usage_sub'] = usage_sub
            results['cvu_total_titles'] = sum(titles_main.values())
            results['cvu_total_usage'] = sum(usage_main.values())
        else:
            results['cvu_available'] = False
    else:
        results['lc_main_counts'] = {}
        results['lc_sub_counts'] = {}
        results['sunburst_data'] = []
        results['cvu_available'] = False

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

        # When a usage column is present, build BOTH title-count and usage-weighted
        # subject distributions so we can render Coverage-vs-Use by subject (this
        # is the analog of the LC version, and it's what makes use metrics visible
        # for vendor reports that have Subject but no LC/call number column —
        # e.g., ProQuest Ebook Central title reports).
        if has_usage_col:
            subj_titles = Counter()      # one count per title-subject occurrence
            subj_usage = Counter()       # weighted by '_weight' (the usage column)
            usage_series = df.loc[idx, '_weight']
            ones_series = pd.Series(1.0, index=idx)
            for ci in range(n_chunks):
                start = ci * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, n_records)
                if start >= n_records:
                    break
                cidx = idx[start:end]
                _profiler_process_subjects_chunk(
                    subj_full.loc[cidx], ones_series.loc[cidx], lc_full.loc[cidx],
                    subj_titles, defaultdict(Counter)
                )
                _profiler_process_subjects_chunk(
                    subj_full.loc[cidx], usage_series.loc[cidx], lc_full.loc[cidx],
                    subj_usage, defaultdict(Counter)
                )
            results['subj_titles'] = subj_titles
            results['subj_usage'] = subj_usage
            results['subj_total_titles'] = sum(subj_titles.values())
            results['subj_total_usage'] = sum(subj_usage.values())
            results['cvu_by_subject_available'] = True
        else:
            results['cvu_by_subject_available'] = False
    else:
        results['subject_counts'] = Counter()
        results['subject_by_lc'] = {}
        results['cvu_by_subject_available'] = False

    progress_bar.progress(80, "Tokenizing titles...")

    # Title-keyword analysis — supplementary lens, NOT merged into subject_counts.
    # Uses uncontrolled vocabulary (title text), so it can surface terminology
    # the controlled subject headings missed. Always built when title column is
    # present; cheap to compute even on large datasets.
    #
    # N-grams: we tokenize each title into segments (split at colons/commas/semis
    # so phrases don't cross subtitle boundaries), drop stopwords, then build
    # all configured n-gram sizes. Each n-gram is stored as a (size, phrase)
    # tuple so the renderer can show separate top-N lists per size.
    if title_col:
        title_keyword_counter = Counter()        # keys: (n, phrase) -> occurrences
        title_keyword_usage = Counter()          # keys: (n, phrase) -> usage sum
        title_unique_titles = Counter()          # keys: (n, phrase) -> distinct titles
        title_series = df.loc[idx, title_col]
        if has_usage_col:
            usage_series = df.loc[idx, '_weight']
        else:
            usage_series = pd.Series(1.0, index=idx)
        ngram_sizes = tuple(sorted(set(int(n) for n in ngram_sizes if int(n) >= 1))) or (1,)
        for ti, raw_title in title_series.items():
            segments = _tokenize_title_segments(raw_title)
            if not segments:
                continue
            u = usage_series.at[ti] if ti in usage_series.index else 0
            seen_in_title = set()
            for n, phrase in _extract_ngrams(segments, n_values=ngram_sizes):
                key = (n, phrase)
                title_keyword_counter[key] += 1
                if has_usage_col:
                    title_keyword_usage[key] += u
                if key not in seen_in_title:
                    title_unique_titles[key] += 1
                    seen_in_title.add(key)
        results['title_keyword_counts'] = title_keyword_counter
        results['title_keyword_usage'] = title_keyword_usage
        results['title_keyword_unique_titles'] = title_unique_titles
        results['title_keyword_available'] = len(title_keyword_counter) > 0
        results['title_keyword_ngram_sizes'] = tuple(ngram_sizes)
    else:
        results['title_keyword_counts'] = Counter()
        results['title_keyword_usage'] = Counter()
        results['title_keyword_unique_titles'] = Counter()
        results['title_keyword_available'] = False
        results['title_keyword_ngram_sizes'] = ()

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


def _classify_signal(ratio, over_thresh, under_thresh, n_titles, min_titles):
    """Return (label, color_hint) for a coverage-vs-use ratio."""
    if n_titles < min_titles:
        return ("—", "gray")
    if ratio >= over_thresh:
        return ("🟢 Overperforming", "green")
    if ratio <= under_thresh:
        return ("🔴 Underperforming", "red")
    return ("✅ Proportional", "neutral")


def _build_cvu_table(titles_dict, usage_dict, total_titles, total_usage,
                     over, under, min_titles, label_lookup, level_col_name):
    """Build the coverage-vs-use dataframe for a given level (main or sub)."""
    all_keys = sorted(set(titles_dict.keys()) | set(usage_dict.keys()))
    rows = []
    for key in all_keys:
        n_titles = titles_dict.get(key, 0)
        n_usage = usage_dict.get(key, 0)
        if n_titles == 0 and n_usage == 0:
            continue
        pct_titles = (n_titles / total_titles * 100) if total_titles > 0 else 0
        pct_usage = (n_usage / total_usage * 100) if total_usage > 0 else 0
        ratio = (pct_usage / pct_titles) if pct_titles > 0 else float('inf') if pct_usage > 0 else 0
        use_per_title = (n_usage / n_titles) if n_titles > 0 else 0
        signal, _ = _classify_signal(ratio, over, under, n_titles, min_titles)
        rows.append({
            level_col_name: key,
            'Description': label_lookup.get(key, '—'),
            'Titles Held': int(n_titles),
            '% of Collection': round(pct_titles, 2),
            'Total Use': int(n_usage),
            '% of Use': round(pct_usage, 2),
            'Use/Title Ratio': round(use_per_title, 2),
            'Use/Holdings Signal': round(ratio, 2) if ratio != float('inf') else None,
            'Assessment': signal,
        })
    return pd.DataFrame(rows)


def _records_drilldown(records_df, key_prefix, *, title_col=None,
                       weight_col=None, author_col=None, location_col=None,
                       has_year=False, has_usage=False, weight_label="Usage",
                       notes="", context_label="", expanded=False):
    """Shared 'show the records behind this' drill-down panel.

    Renders an expander containing the underlying records for whatever
    aggregate the caller has already filtered to (an LC range, a subclass, a
    subject term, etc.). The caller pre-filters; this helper handles the rich
    secondary refinement that's identical everywhere:

      - usage threshold slider (when a usage column exists)
      - year range filter (when a _year column exists)
      - sort control (by usage, title, author, call number, year)
      - column visibility toggle
      - record count + sortable table + CSV export to the download tray

    Args:
      records_df:    already-scoped slice (rows for the clicked aggregate).
                     Expected to carry _weight / _year / _lc_sub / _lc_number /
                     _lc_range where available.
      key_prefix:    unique widget-key prefix so multiple drill-downs coexist
                     on the same page without key collisions.
      context_label: human-readable description of the scope (e.g.
                     "HQ 1101–2030.7 — Women, feminism" or "Subject: sociology").
                     Shown in the expander header and baked into export metadata.
      expanded:      whether the expander starts open (default False).
    """
    n_records = len(records_df)
    header = f"🔎 Show the {n_records:,} record{'s' if n_records != 1 else ''} behind this"
    if context_label:
        header += f"  ·  {context_label}"

    with st.expander(header, expanded=expanded):
        if n_records == 0:
            st.caption("No records match this selection.")
            return

        work = records_df.copy()

        # --- Secondary refinement controls ---
        # Lay them across columns so the panel stays compact.
        ctrl_cols = st.columns(3)

        # (1) Usage threshold — only meaningful with a usage column
        with ctrl_cols[0]:
            if has_usage and weight_col and '_weight' in work.columns:
                wmax = int(work['_weight'].max()) if n_records else 0
                if wmax > 0:
                    mode = st.selectbox(
                        f"{weight_label} filter",
                        ["All", "Zero only", "At or below…", "At or above…"],
                        key=f"{key_prefix}_umode",
                    )
                    if mode == "Zero only":
                        work = work[work['_weight'] <= 0]
                    elif mode in ("At or below…", "At or above…"):
                        thr = st.number_input(
                            f"{weight_label} threshold",
                            min_value=0, max_value=wmax,
                            value=0 if mode == "At or below…" else wmax,
                            key=f"{key_prefix}_uthr",
                        )
                        if mode == "At or below…":
                            work = work[work['_weight'] <= thr]
                        else:
                            work = work[work['_weight'] >= thr]
                else:
                    st.caption("No usage values to filter.")
            else:
                st.caption("No usage column.")

        # (2) Year range — only when a _year column exists
        with ctrl_cols[1]:
            if has_year and '_year' in work.columns:
                yrs = sorted(int(y) for y in work['_year'].dropna().unique())
                if len(yrs) > 1:
                    y_lo, y_hi = st.select_slider(
                        "Year range",
                        options=yrs,
                        value=(yrs[0], yrs[-1]),
                        key=f"{key_prefix}_yrs",
                    )
                    work = work[(work['_year'] >= y_lo) & (work['_year'] <= y_hi)]
                elif len(yrs) == 1:
                    st.caption(f"All from {yrs[0]}.")
                else:
                    st.caption("No year data.")
            else:
                st.caption("No year column.")

        # (3) Sort control
        with ctrl_cols[2]:
            sort_opts = []
            if has_usage and '_weight' in work.columns:
                sort_opts.append(f"{weight_label} (high→low)")
                sort_opts.append(f"{weight_label} (low→high)")
            if title_col and title_col in work.columns:
                sort_opts.append("Title (A→Z)")
            if author_col and author_col in work.columns:
                sort_opts.append("Author (A→Z)")
            if '_lc_number' in work.columns:
                sort_opts.append("Call number")
            if has_year and '_year' in work.columns:
                sort_opts.append("Year (newest)")
                sort_opts.append("Year (oldest)")
            sort_opts = sort_opts or ["(none)"]
            sort_choice = st.selectbox("Sort by", sort_opts, key=f"{key_prefix}_sort")

        # Apply sort
        if sort_choice.startswith(weight_label) and '_weight' in work.columns:
            work = work.sort_values('_weight', ascending="low→high" in sort_choice)
        elif sort_choice == "Title (A→Z)" and title_col in work.columns:
            work = work.sort_values(title_col)
        elif sort_choice == "Author (A→Z)" and author_col and author_col in work.columns:
            work = work.sort_values(author_col)
        elif sort_choice == "Call number":
            sort_keys = [c for c in ['_lc_sub', '_lc_number'] if c in work.columns]
            if sort_keys:
                work = work.sort_values(sort_keys)
        elif sort_choice.startswith("Year"):
            work = work.sort_values('_year', ascending="oldest" in sort_choice)

        # --- Build the display frame: friendly columns only ---
        display_map = []
        if title_col and title_col in work.columns:
            display_map.append((title_col, 'Title'))
        if author_col and author_col in work.columns:
            display_map.append((author_col, 'Author'))
        if '_lc_sub' in work.columns:
            display_map.append(('_lc_sub', 'LC Subclass'))
        if '_lc_number' in work.columns:
            display_map.append(('_lc_number', 'LC Number'))
        if '_lc_range' in work.columns:
            display_map.append(('_lc_range', 'LC Range'))
        if location_col and location_col in work.columns:
            display_map.append((location_col, 'Location'))
        if has_year and '_year' in work.columns:
            display_map.append(('_year', 'Year'))
        if has_usage and '_weight' in work.columns:
            display_map.append(('_weight', weight_label))

        if not display_map:
            st.caption("No displayable columns.")
            return

        # Optional column visibility toggle
        all_display_names = [name for _, name in display_map]
        chosen = st.multiselect(
            "Columns to show",
            options=all_display_names,
            default=all_display_names,
            key=f"{key_prefix}_cols",
        )
        if not chosen:
            chosen = all_display_names  # never show an empty table

        view = work[[src for src, name in display_map if name in chosen]].copy()
        view.columns = [name for _, name in display_map if name in chosen]

        # Tidy the year column (no decimals) and LC number
        if 'Year' in view.columns:
            view['Year'] = view['Year'].astype('Int64')
        if 'LC Number' in view.columns:
            view['LC Number'] = view['LC Number'].round(2)

        st.caption(f"Showing {len(view):,} record{'s' if len(view) != 1 else ''} "
                   f"after refinement.")
        st.dataframe(view, use_container_width=True, hide_index=True, height=360)

        # Export — to the profiler tray + a direct button
        safe_ctx = (context_label or "records").replace(' ', '_').replace('/', '-')
        safe_ctx = ''.join(ch for ch in safe_ctx if ch.isalnum() or ch in '_-.')[:60]
        fname = f"records_{safe_ctx}.csv"
        _rec_bytes = _annotate_csv(
            view, notes,
            extra_meta={'Tool': 'Collection Profiler',
                        'View': 'Records drill-down',
                        'Scope': context_label or '(unscoped)',
                        'Records': len(view)}
        )
        st.download_button(
            "📥 These records (CSV)", _rec_bytes, fname, "text/csv",
            key=f"{key_prefix}_dl",
        )
        _add_to_tray("profiler", fname, _rec_bytes)


def _render_coverage_vs_use_by_subject(results, settings, notes="", records_ctx=None):
    """Coverage vs. Use broken down by Subject term.

    Used when the file has a usage column but no LC/call number (e.g., ProQuest
    Ebook Central title reports). Mirrors the LC version's logic and thresholds.
    """
    st.markdown("---")
    st.subheader("Coverage vs. use — by subject")
    st.markdown(
        "Compares **% of titles** carrying each subject against **% of use** those titles drive. "
        "Useful when your file has subject headings but no LC/call number column."
    )
    st.markdown(
        f"- 🟢 **Overperforming** — ratio ≥ **{settings['cvu_over']}** (subject area pulling heavy use; consider expanding)\n"
        f"- 🔴 **Underperforming** — ratio ≤ **{settings['cvu_under']}** (well-represented but lightly used; review or weed)\n"
        f"- ✅ **Proportional** — use roughly matches representation\n"
        f"- — **Insufficient data** — fewer than **{settings['cvu_min_titles']}** title-tags in that subject"
    )
    st.caption(
        "Note: a single title with multiple subjects contributes to each. "
        "Totals here count *title-subject pairs*, not unique titles."
    )

    usage_label = settings.get('usage_col_label', 'Usage')
    titles_dict = results['subj_titles']
    usage_dict = results['subj_usage']
    total_titles = results['subj_total_titles']
    total_usage = results['subj_total_usage']
    top_n = settings.get('top_n_subjects', 30)

    # KPI summary
    k1, k2, k3 = st.columns(3)
    k1.metric("Subject-Title Tags", f"{int(total_titles):,}")
    k2.metric(f"Total {usage_label}", f"{int(total_usage):,}")
    k3.metric(f"Overall {usage_label} / Tag",
              f"{total_usage / max(1, total_titles):.2f}")

    if total_usage == 0:
        st.warning(
            f"⚠️ The selected usage column has **zero total {usage_label.lower()}** across "
            "all titles in this file. Coverage-vs-Use analysis needs at least some non-zero "
            "values. Double-check that you mapped the right column above."
        )
        return

    subj_df = _build_cvu_table(
        titles_dict, usage_dict,
        total_titles, total_usage,
        settings['cvu_over'], settings['cvu_under'], settings['cvu_min_titles'],
        {}, 'Subject'
    )
    if subj_df.empty:
        st.info("No subject data to display.")
        return

    # Drop the 'Description' column — not meaningful for subject terms
    if 'Description' in subj_df.columns:
        subj_df = subj_df.drop(columns='Description')

    # Sort by Total Use descending so the heaviest-used subjects lead, then trim
    subj_df = subj_df.sort_values('Total Use', ascending=False).head(top_n)

    st.dataframe(
        subj_df, use_container_width=True, hide_index=True, height=min(450, 50 + 35 * len(subj_df)),
        column_config={
            '% of Collection': st.column_config.NumberColumn(format="%.2f%%"),
            '% of Use': st.column_config.NumberColumn(format="%.2f%%"),
            'Use/Holdings Signal': st.column_config.NumberColumn(
                format="%.2f",
                help="(% of use) ÷ (% of representation). 1.0 = proportional."
            ),
        }
    )

    _bytes = _annotate_csv(subj_df, notes,
                           extra_meta={'Tool': 'Collection Profiler',
                                       'View': 'Coverage vs. Use by Subject',
                                       'Usage column': usage_label,
                                       'Over threshold': settings['cvu_over'],
                                       'Under threshold': settings['cvu_under']})
    st.download_button(f"📥 Coverage-vs-Use by Subject (top {top_n}) CSV",
                       _bytes,
                       "coverage_vs_use_by_subject.csv",
                       "text/csv", key='prof_dl_cvu_subj')
    _add_to_tray("profiler", "coverage_vs_use_by_subject.csv", _bytes)

    # --- Drill-down: records tagged with a chosen subject ---
    if records_ctx is not None and records_ctx.get('subject_col'):
        _rdf = records_ctx['df']
        scol = records_ctx['subject_col']
        if scol in _rdf.columns:
            subj_options = subj_df['Subject'].dropna().tolist()
            if subj_options:
                pick_subj = st.selectbox(
                    "🔎 Inspect the records tagged with a subject",
                    options=["(choose a subject)"] + subj_options,
                    key="prof_cvu_subj_drill_pick",
                    help="Records whose subject field contains the chosen term "
                         "(case-insensitive substring match).",
                )
                if pick_subj != "(choose a subject)":
                    # Substring match against the raw subject column. The CVU
                    # table's subject terms come from normalized/split subjects,
                    # so we match case-insensitively on the contained term.
                    mask = _rdf[scol].astype(str).str.contains(
                        re.escape(pick_subj), case=False, na=False
                    )
                    scope = _rdf[mask]
                    _records_drilldown(
                        scope, key_prefix=f"cvu_subj_{abs(hash(pick_subj)) % 100000}",
                        title_col=records_ctx.get('title_col'),
                        weight_col=records_ctx.get('weight_col'),
                        author_col=records_ctx.get('author_col'),
                        location_col=records_ctx.get('location_col'),
                        has_year=records_ctx.get('has_year', False),
                        has_usage=records_ctx.get('has_usage', False),
                        weight_label=records_ctx.get('weight_label', 'Usage'),
                        notes=notes,
                        context_label=f"Subject: {pick_subj}",
                        expanded=True,
                    )


def _render_coverage_vs_use(results, settings, notes="", records_ctx=None):
    """Render the Coverage vs. Use section — the core 'what we have vs what's used' view.

    records_ctx (optional dict) enables 'show the records behind this' drill-downs.
    Expected keys: df, title_col, weight_col, author_col, location_col,
    has_year, has_usage, weight_label.
    """
    st.markdown("---")
    st.subheader("Coverage vs. use")
    st.markdown(
        "Compares **% of your collection** in each LC area against **% of use** it drives. "
        "The *Assessment* column uses ratios you can tune in the sidebar:"
    )
    st.markdown(
        f"- 🟢 **Overperforming** — ratio ≥ **{settings['cvu_over']}** (small areas pulling heavy use; candidates for expansion)\n"
        f"- 🔴 **Underperforming** — ratio ≤ **{settings['cvu_under']}** (large areas with thin use; candidates for weeding or reassessment)\n"
        f"- ✅ **Proportional** — use roughly matches holdings\n"
        f"- — **Insufficient data** — fewer than **{settings['cvu_min_titles']}** titles in that area"
    )

    usage_label = settings.get('usage_col_label', 'Usage')
    total_titles = results['cvu_total_titles']
    total_usage = results['cvu_total_usage']

    over = settings['cvu_over']
    under = settings['cvu_under']
    min_titles = settings['cvu_min_titles']

    # KPI summary
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Titles", f"{int(total_titles):,}")
    k2.metric(f"Total {usage_label}", f"{int(total_usage):,}")
    k3.metric("Overall Use / Title",
              f"{total_usage / max(1, total_titles):.2f}")

    # ---- LC main class table & chart ----
    st.markdown("#### LC Main Class")
    main_df = _build_cvu_table(
        results['cvu_titles_main'], results['cvu_usage_main'],
        total_titles, total_usage,
        over, under, min_titles,
        LC_CLASSES, 'LC Class'
    )
    if main_df.empty:
        st.info("No LC data to display.")
        return

    # Sort by signal ratio descending (so overperformers appear first)
    # But put "—" (insufficient data) at the bottom
    main_df['_sort_key'] = main_df['Use/Holdings Signal'].fillna(-1)
    main_df = main_df.sort_values('_sort_key', ascending=False).drop(columns='_sort_key')

    st.dataframe(
        main_df, use_container_width=True, hide_index=True, height=400,
        column_config={
            '% of Collection': st.column_config.NumberColumn(format="%.2f%%"),
            '% of Use': st.column_config.NumberColumn(format="%.2f%%"),
            'Use/Holdings Signal': st.column_config.NumberColumn(
                format="%.2f",
                help="(% of use) ÷ (% of holdings). 1.0 = proportional."
            ),
        }
    )

    _cvu_main_bytes = _annotate_csv(main_df, notes,
                                    extra_meta={'Tool': 'Collection Profiler',
                                                'View': 'Coverage vs. Use (main)',
                                                'Over threshold': settings['cvu_over'],
                                                'Under threshold': settings['cvu_under']})
    st.download_button("📥 Coverage-vs-Use (Main Class) CSV",
                       _cvu_main_bytes,
                       "coverage_vs_use_main.csv", "text/csv",
                       key='prof_dl_cvu_main')
    _add_to_tray("profiler", "coverage_vs_use_main.csv", _cvu_main_bytes)

    # --- Drill-down: records behind a chosen LC main class ---
    if records_ctx is not None and '_lc_main' in records_ctx['df'].columns:
        _rdf = records_ctx['df']
        # Offer the flagged (over/under) classes first since those are the
        # ones a user most wants to audit, then any class with data.
        flagged_first = main_df.copy()
        flagged_first['_rank'] = flagged_first['Assessment'].map(
            {"🟢 Overperforming": 0, "🔴 Underperforming": 1,
             "✅ Proportional": 2, "—": 3}
        ).fillna(4)
        ordered_classes = (flagged_first.sort_values(['_rank'])
                           ['LC Class'].tolist())
        if ordered_classes:
            pick = st.selectbox(
                "🔎 Inspect the records behind a main class",
                options=["(choose a class)"] + ordered_classes,
                key="prof_cvu_main_drill_pick",
                help="Pick an LC main class to see the underlying titles — "
                     "useful for auditing why a class is flagged over- or "
                     "underperforming.",
            )
            if pick != "(choose a class)":
                scope = _rdf[_rdf['_lc_main'] == pick]
                row = main_df[main_df['LC Class'] == pick]
                assess = row['Assessment'].iloc[0] if not row.empty else ""
                ctx_label = f"LC {pick} – {LC_CLASSES.get(pick, '?')} {assess}".strip()
                _records_drilldown(
                    scope, key_prefix=f"cvu_main_{pick}",
                    title_col=records_ctx.get('title_col'),
                    weight_col=records_ctx.get('weight_col'),
                    author_col=records_ctx.get('author_col'),
                    location_col=records_ctx.get('location_col'),
                    has_year=records_ctx.get('has_year', False),
                    has_usage=records_ctx.get('has_usage', False),
                    weight_label=records_ctx.get('weight_label', 'Usage'),
                    notes=notes, context_label=ctx_label, expanded=True,
                )

    # Scatter plot: % titles vs % use, with diagonal reference line
    plot_df = main_df[main_df['Assessment'] != "—"].copy()
    if not plot_df.empty:
        fig = px.scatter(
            plot_df, x='% of Collection', y='% of Use',
            size='Titles Held', color='Assessment',
            color_discrete_map={
                "🟢 Overperforming": "#2ecc71",
                "🔴 Underperforming": "#e74c3c",
                "✅ Proportional": "#71C5E8",
            },
            hover_data=['LC Class', 'Description', 'Total Use', 'Use/Title Ratio'],
            text='LC Class',
            title="Collection Coverage vs. Use (by LC Main Class)",
        )
        # Diagonal reference: if use were perfectly proportional to holdings
        max_val = max(plot_df['% of Collection'].max(), plot_df['% of Use'].max()) * 1.1
        fig.add_shape(
            type='line', line=dict(color='gray', dash='dash', width=1),
            x0=0, y0=0, x1=max_val, y1=max_val,
        )
        fig.add_annotation(
            x=max_val * 0.9, y=max_val * 0.95,
            text="Proportional line", showarrow=False,
            font=dict(size=10, color="gray"),
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500, xaxis_title="% of Collection",
                          yaxis_title=f"% of {usage_label}")
        st.plotly_chart(fig, use_container_width=True)

    # ---- LC subclass table (optional) ----
    if settings.get('cvu_show_sub') and results.get('cvu_titles_sub'):
        st.markdown("#### LC Subclass")
        st.caption("Drill into the same comparison at the subclass level "
                   "(e.g., HQ1000s vs. HQ750s).")

        # Build a flat subclass lookup from LC_SUBCLASSES dict
        sub_lookup = {}
        for main_class, subs in LC_SUBCLASSES.items():
            sub_lookup.update(subs)

        sub_df = _build_cvu_table(
            results['cvu_titles_sub'], results['cvu_usage_sub'],
            total_titles, total_usage,
            over, under, min_titles,
            sub_lookup, 'LC Subclass'
        )
        if sub_df.empty:
            st.info("No LC subclass data to display.")
        else:
            # Allow filtering to just one main class for readability
            main_classes_present = sorted(set(
                sc[0] for sc in sub_df['LC Subclass'] if isinstance(sc, str) and sc
            ))
            filter_main = st.selectbox(
                "Filter subclass view by main class (optional)",
                options=["All"] + [f"{c} – {LC_CLASSES.get(c, '?')}"
                                    for c in main_classes_present],
                key='prof_cvu_sub_filter'
            )
            if filter_main != "All":
                prefix = filter_main.split(' –')[0]
                sub_df_display = sub_df[
                    sub_df['LC Subclass'].str.startswith(prefix, na=False)
                ].copy()
            else:
                sub_df_display = sub_df.copy()

            sub_df_display['_sort_key'] = sub_df_display['Use/Holdings Signal'].fillna(-1)
            sub_df_display = sub_df_display.sort_values(
                '_sort_key', ascending=False
            ).drop(columns='_sort_key')

            st.dataframe(
                sub_df_display, use_container_width=True,
                hide_index=True, height=400,
                column_config={
                    '% of Collection': st.column_config.NumberColumn(format="%.2f%%"),
                    '% of Use': st.column_config.NumberColumn(format="%.2f%%"),
                    'Use/Holdings Signal': st.column_config.NumberColumn(format="%.2f"),
                }
            )
            _cvu_sub_bytes = _annotate_csv(sub_df, notes,
                                           extra_meta={'Tool': 'Collection Profiler',
                                                       'View': 'Coverage vs. Use (subclass)'})
            st.download_button("📥 Coverage-vs-Use (Subclass) CSV",
                               _cvu_sub_bytes,
                               "coverage_vs_use_subclass.csv", "text/csv",
                               key='prof_dl_cvu_sub')
            _add_to_tray("profiler", "coverage_vs_use_subclass.csv", _cvu_sub_bytes)

            # --- Drill-down: records behind a chosen subclass ---
            if records_ctx is not None and '_lc_sub' in records_ctx['df'].columns:
                _rdf = records_ctx['df']
                sub_ordered = (sub_df_display['LC Subclass']
                               .dropna().tolist())
                if sub_ordered:
                    pick_sub = st.selectbox(
                        "🔎 Inspect the records behind a subclass",
                        options=["(choose a subclass)"] + sub_ordered,
                        key="prof_cvu_sub_drill_pick",
                    )
                    if pick_sub != "(choose a subclass)":
                        scope = _rdf[_rdf['_lc_sub'] == pick_sub]
                        _records_drilldown(
                            scope, key_prefix=f"cvu_sub_{pick_sub}",
                            title_col=records_ctx.get('title_col'),
                            weight_col=records_ctx.get('weight_col'),
                            author_col=records_ctx.get('author_col'),
                            location_col=records_ctx.get('location_col'),
                            has_year=records_ctx.get('has_year', False),
                            has_usage=records_ctx.get('has_usage', False),
                            weight_label=records_ctx.get('weight_label', 'Usage'),
                            notes=notes,
                            context_label=f"Subclass {pick_sub}",
                            expanded=True,
                        )

    # Interpretive callout
    over_count = (main_df['Assessment'] == "🟢 Overperforming").sum()
    under_count = (main_df['Assessment'] == "🔴 Underperforming").sum()
    if over_count or under_count:
        st.markdown("**What to do with this:**")
        bullets = []
        if over_count:
            bullets.append(
                f"- **{over_count} overperforming area(s)** are candidates for "
                "deeper investment — strong use suggests demand you could grow into."
            )
        if under_count:
            bullets.append(
                f"- **{under_count} underperforming area(s)** are candidates for "
                "weeding review or reassessment. Switch to the **Title Analysis** "
                "tab to see the specific low-use titles driving those numbers."
            )
        st.markdown("\n".join(bullets))


def _render_title_keywords(results, settings, notes=""):
    """Render the title-keyword view: bar chart + optional word cloud + table.

    This is a SEPARATE lens from the subject view. It tokenizes title text
    (uncontrolled vocabulary) with stopwords stripped, so it surfaces
    terminology that subject headings might have missed — newer concepts,
    methodological terms, interdisciplinary phrases. Distinct from subject
    analysis on purpose: titles are not curated metadata.
    """
    if not results.get('title_keyword_available'):
        return

    counter = results['title_keyword_counts']
    usage_counter = results.get('title_keyword_usage', Counter())
    unique_titles = results.get('title_keyword_unique_titles', Counter())
    available_sizes = results.get('title_keyword_ngram_sizes', (1,))

    # Apply user-tunable extra stopwords. Stopwords are applied per-token so
    # bigrams/trigrams containing any user-flagged word also get filtered.
    extra_stops = settings.get('tk_extra_stopwords') or set()

    def _key_passes_stops(key):
        # key is (n, phrase). Drop the whole n-gram if any token is in extra_stops.
        if not extra_stops:
            return True
        _, phrase = key
        return not any(tok in extra_stops for tok in phrase.split())

    if extra_stops:
        counter = Counter({k: v for k, v in counter.items() if _key_passes_stops(k)})
        usage_counter = Counter({k: v for k, v in usage_counter.items() if _key_passes_stops(k)})
        unique_titles = Counter({k: v for k, v in unique_titles.items() if _key_passes_stops(k)})

    if not counter:
        return

    top_n = settings.get('tk_top_n', 30)
    has_usage = bool(settings.get('has_usage_col'))
    usage_label = settings.get('usage_col_label', 'Usage')
    selected_sizes = tuple(settings.get('tk_ngram_sizes', (1, 2, 3)))
    min_freq = settings.get('tk_min_freq', 2)

    # Restrict to sizes the user actually selected AND that we built
    sizes_to_show = tuple(n for n in selected_sizes if n in available_sizes)
    if not sizes_to_show:
        sizes_to_show = available_sizes

    st.markdown("---")
    st.subheader("Title keywords & phrases")
    st.caption(
        "A supplementary lens on the collection. Unlike the subject view above "
        "(which uses curated subject headings), this counts words and phrases "
        "appearing in **title text** with English stopwords and common "
        "publishing-noise words removed. N-grams are built from tokens that "
        "survive stopword removal and never cross subtitle punctuation. "
        "Use this view to spot terminology that controlled subject vocabularies "
        "may have missed — newer concepts, methodological terms, "
        "interdisciplinary phrases. Treat findings as exploratory."
    )

    _SIZE_LABELS = {
        1: "Single words",
        2: "Two-word phrases",
        3: "Three-word phrases",
    }
    tab_labels = [f"{_SIZE_LABELS.get(n, f'{n}-grams')} (top {top_n})"
                  for n in sizes_to_show]
    tabs = st.tabs(tab_labels) if len(sizes_to_show) > 1 else [None]

    all_export_rows = []  # for combined CSV export at end

    for tab, n in zip(tabs, sizes_to_show):
        ctx = tab if tab is not None else st.container()
        with ctx:
            # Filter: for n>1, require the phrase to appear in at least `min_freq`
            # distinct titles (not just total occurrences — this filters out
            # duplicate-title noise like a book listed under multiple ISBNs).
            # Unigrams always passed (they're already filtered by min word length).
            this_min = 1 if n == 1 else min_freq
            sub_items = []
            for key, cnt in counter.items():
                if key[0] != n:
                    continue
                ut = unique_titles.get(key, 0)
                if n > 1 and ut < this_min:
                    continue
                sub_items.append((key[1], cnt))
            if not sub_items:
                st.info(
                    f"No {_SIZE_LABELS.get(n, str(n)+'-gram').lower()} appeared in "
                    f"at least {this_min} distinct title{'s' if this_min != 1 else ''}. "
                    "Try lowering **Min occurrences** in the title-keyword options "
                    "(it filters by distinct titles for multi-word phrases)."
                )
                continue
            sub_items.sort(key=lambda x: -x[1])
            top_items = sub_items[:top_n]

            rows = []
            for phrase, occurrences in top_items:
                key = (n, phrase)
                row = {
                    'Phrase' if n > 1 else 'Keyword': phrase,
                    'Title Occurrences': int(occurrences),
                    'Distinct Titles': int(unique_titles.get(key, 0)),
                }
                if has_usage:
                    row[f'Total {usage_label}'] = int(usage_counter.get(key, 0))
                rows.append(row)
                all_export_rows.append({
                    'N-gram Size': n,
                    'Phrase': phrase,
                    'Title Occurrences': int(occurrences),
                    'Distinct Titles': int(unique_titles.get(key, 0)),
                    f'Total {usage_label}': int(usage_counter.get(key, 0)) if has_usage else None,
                })
            kw_df = pd.DataFrame(rows)
            label_col = 'Phrase' if n > 1 else 'Keyword'

            fig = px.bar(
                kw_df, x='Title Occurrences', y=label_col,
                orientation='h', color='Title Occurrences',
                color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                hover_data=[c for c in ['Distinct Titles', f'Total {usage_label}']
                            if c in kw_df.columns],
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(450, len(kw_df) * 24),
                showlegend=False,
                margin=dict(t=30, l=0, r=0, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"prof_tk_chart_{n}")

            with st.expander(f"📋 Full {label_col.lower()} table (sortable)"):
                st.dataframe(
                    kw_df, use_container_width=True, hide_index=True,
                    height=min(450, 50 + 35 * len(kw_df)),
                )

    # Single combined CSV export covering all selected sizes
    if all_export_rows:
        export_df = pd.DataFrame(all_export_rows)
        if not has_usage and f'Total {usage_label}' in export_df.columns:
            export_df = export_df.drop(columns=[f'Total {usage_label}'])
        _kw_bytes = _annotate_csv(
            export_df, notes,
            extra_meta={'Tool': 'Collection Profiler',
                        'View': 'Title Keywords (n-grams)',
                        'Top N (per size)': top_n,
                        'N-gram sizes': ', '.join(str(n) for n in sizes_to_show),
                        'Min occurrences (n>1)': min_freq,
                        'Stopwords': 'Built-in English + library/publishing noise'
                                     + (f' + {len(extra_stops)} custom' if extra_stops else '')}
        )
        st.download_button(
            "📥 Title keywords & phrases (CSV)", _kw_bytes,
            "title_keywords.csv", "text/csv", key='prof_dl_tk',
        )
        _add_to_tray("profiler", "title_keywords.csv", _kw_bytes)

    # Word cloud — unigrams only (clouds don't render multi-word phrases well).
    if settings.get('tk_show_wordcloud') and WORDCLOUD_AVAILABLE:
        unigram_counter = Counter({key[1]: cnt for key, cnt in counter.items()
                                   if key[0] == 1})
        if not unigram_counter:
            return
        st.markdown("##### Title keyword cloud (single words)")
        max_words = settings.get('tk_wc_max_words', 100)
        color_scheme = settings.get('tk_wc_color', 'plasma')
        cloud_data = dict(unigram_counter.most_common(max_words))
        if cloud_data:
            wc = WordCloud(
                width=1200, height=500, background_color='white',
                colormap=color_scheme, max_words=max_words,
                relative_scaling=0.5, min_font_size=10, prefer_horizontal=0.7,
            ).generate_from_frequencies(cloud_data)
            fig_wc, ax_wc = plt.subplots(figsize=(14, 6))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc, use_container_width=True)
            buf = BytesIO()
            fig_wc.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            buf.seek(0)
            _tk_wc_bytes = buf.getvalue()
            st.download_button(
                "📥 Title keyword cloud (PNG)", _tk_wc_bytes,
                "title_keywords_wordcloud.png", "image/png",
                key='prof_dl_tk_wc',
            )
            _add_to_tray("profiler", "title_keywords_wordcloud.png", _tk_wc_bytes)
            plt.close(fig_wc)
    elif settings.get('tk_show_wordcloud') and not WORDCLOUD_AVAILABLE:
        st.info("Install `wordcloud` and `matplotlib` to enable the keyword cloud.")


def _profiler_display_results(results, settings, df, idx,
                              title_col=None, weight_col=None,
                              author_col=None, date_col=None,
                              location_col=None, subj_col=None,
                              lc_col=None):
    """Render Profiler results in three top-level tabs: LC Analysis, Subject
    Term Analysis, and Title Analysis.

    The Title Analysis tab absorbs the functionality formerly in the standalone
    Print/Usage Analyzer: top titles by usage, weeding review (low-usage threshold),
    LC breakdown by circulation, author summary, date-range filter. These are
    enabled when the file has a usage column.

    Args beyond `results`/`settings`/`df`/`idx` are the detected column names from
    the upstream loader. They're optional because not every file has all of them.
    """
    wl = settings['weight_label']
    top_n = settings['top_n_subjects']
    usage_col_label = settings.get('usage_col_label', 'Usage')
    has_usage = bool(settings.get('has_usage_col'))

    # Fresh tray for this render pass
    _reset_tray("profiler")

    # ---- Overview KPIs (shown above tabs so they're always visible) ----
    st.markdown("---")
    st.subheader("Collection overview")
    total_use = 0
    if has_usage:
        if results.get('cvu_available'):
            total_use = results.get('cvu_total_usage', 0)
        elif results.get('cvu_by_subject_available'):
            total_use = results.get('subj_total_usage', 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records Analyzed", f"{results['n_records']:,}")
    c2.metric(f"Total {wl}", f"{results['total_weight']:,.0f}")
    if has_usage and wl != usage_col_label:
        c3.metric(f"Total {usage_col_label}", f"{int(total_use):,}")
        c4.metric("Unique Subjects", f"{len(results['subject_counts']):,}")
    else:
        c3.metric("LC Classes Present", f"{len(results['lc_main_counts'])}")
        c4.metric("Unique Subjects", f"{len(results['subject_counts']):,}")

    # ---- Shared analysis-notes widget (used across all tabs) ----
    notes = _notes_widget(
        "profiler",
        placeholder="e.g., Prepared for sociology accreditation report, Nov 2025. "
                    "Follow-up: discuss HQ underperformance with Dr. Chen."
    )

    # ---- Shared records context for 'show the records behind this' drill-downs ----
    # Built once here so every analytical view can offer a drill-down into the
    # underlying titles. The records frame is the filtered view (df.loc[idx]),
    # augmented with a _year column when a date/year column was detected so the
    # drill-down's year filter works. All of _lc_sub / _lc_number / _lc_range /
    # _weight are already present from upstream processing.
    _records_ctx = None
    if df is not None:
        _rec_df = df.loc[idx] if idx is not None else df
        _ctx_has_year = False
        if date_col and date_col in _rec_df.columns:
            _rec_df = _rec_df.copy()
            _src = _rec_df[date_col]
            if pd.api.types.is_numeric_dtype(_src):
                _rec_df['_year'] = pd.to_numeric(_src, errors='coerce').astype('Int64')
            else:
                _parsed = pd.to_datetime(_src, errors='coerce')
                _rec_df['_year'] = _parsed.dt.year.astype('Int64')
            _ctx_has_year = _rec_df['_year'].notna().any()
        _records_ctx = {
            'df': _rec_df,
            'title_col': title_col,
            'weight_col': weight_col,
            'author_col': author_col,
            'location_col': location_col,
            'subject_col': subj_col,
            'has_year': bool(_ctx_has_year),
            'has_usage': bool(has_usage),
            'weight_label': usage_col_label if has_usage else wl,
        }

    # ---- Three top-level tabs ----
    st.markdown("---")
    tab_lc, tab_subj, tab_title = st.tabs([
        "🗺️ LC Analysis",
        "🏷️ Subject Term Analysis",
        "📚 Title Analysis",
    ])

    # ======================================================================
    # TAB 1: LC ANALYSIS
    # ======================================================================
    with tab_lc:
        if not results.get('lc_main_counts'):
            st.info("No LC classification data found. Make sure the LC column "
                    "is mapped correctly, or upload a file with LC call numbers.")
        else:
            # --- LC sunburst ---
            if settings['show_sunburst'] and results['sunburst_data']:
                st.subheader("LC classification sunburst")
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

            # --- LC treemap ---
            if settings['show_treemap'] and results['lc_main_counts']:
                st.markdown("---")
                st.subheader("LC classification treemap")
                tm_data = [{'Class': f"{c} – {LC_CLASSES.get(c, c)}", 'Count': ct,
                            'Pct': ct / results['total_weight'] * 100}
                           for c, ct in sorted(results['lc_main_counts'].items(), key=lambda x: -x[1])]
                tm_df = pd.DataFrame(tm_data)
                fig = px.treemap(tm_df, path=['Class'], values='Count', color='Count',
                                 color_continuous_scale=[[0, '#71C5E8'], [0.5, '#285C4D'], [1, '#1a3d33']],
                                 hover_data={'Pct': ':.1f'})
                fig.update_layout(height=500, margin=dict(t=30, l=0, r=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

            # --- LC × Subject heatmap (inherently cross-axis; living in LC tab) ---
            if settings['show_heatmap'] and results['subject_by_lc']:
                st.markdown("---")
                st.subheader("LC class × top subjects heatmap")
                st.caption("Bridges LC and subject analysis: shows which subject "
                           "terms cluster within which LC classes.")
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

            # --- Coverage vs. Use (LC-based, when LC + usage are both present) ---
            if settings.get('show_coverage_vs_use') and results.get('cvu_available'):
                st.markdown("---")
                _render_coverage_vs_use(results, settings, notes=notes,
                                        records_ctx=_records_ctx)

            # --- Gap analysis ---
            if settings['show_gap_analysis']:
                st.markdown("---")
                st.subheader("Collection gap analysis")
                missing = results['missing_lc_classes']
                thin = results['thin_lc_classes']
                if missing:
                    st.markdown("**LC Classes with No Holdings:**")
                    st.dataframe(pd.DataFrame([{'LC Class': c, 'Description': LC_CLASSES.get(c, '')}
                                               for c in missing]),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("All LC main classes are represented.")
                if thin:
                    st.markdown("**LC Classes Below 1% of Collection:**")
                    rows_t = [{'LC Class': c, 'Description': LC_CLASSES.get(c, ''),
                               f'{wl}': f"{v:,.0f}",
                               '% of Collection': f"{v / results['total_weight'] * 100:.2f}%"}
                              for c, v in sorted(thin.items(), key=lambda x: x[1])]
                    st.dataframe(pd.DataFrame(rows_t),
                                 use_container_width=True, hide_index=True)
                if results['lc_main_counts']:
                    st.markdown("**Strongest Areas (top 5):**")
                    top5 = sorted(results['lc_main_counts'].items(), key=lambda x: -x[1])[:5]
                    rows_s = [{'LC Class': c, 'Description': LC_CLASSES.get(c, ''),
                               f'{wl}': f"{v:,.0f}",
                               '% of Collection': f"{v / results['total_weight'] * 100:.1f}%"}
                              for c, v in top5]
                    st.dataframe(pd.DataFrame(rows_s),
                                 use_container_width=True, hide_index=True)

            # --- Sub-class range distribution ---
            # Drills inside selected LC subclasses (HQ, PR, PS, etc.) to show
            # which numeric ranges are getting the most representation/use.
            # The range labels come from LC_RANGES (curated where high-traffic,
            # bucketed by hundreds otherwise via lookup_lc_range).
            if df is not None and '_lc_range' in df.columns and '_lc_sub' in df.columns:
                df_lc = df.loc[idx] if idx is not None else df
                # Filter to rows with a parseable subclass
                df_lc = df_lc[df_lc['_lc_sub'].notna() & df_lc['_lc_range'].notna()]
                if len(df_lc) > 0:
                    st.markdown("---")
                    st.subheader("Sub-class range distribution")
                    st.caption("Drills below the two-letter subclass to show "
                               "which LC ranges within it are most represented "
                               "or most used. Choose one or more subclasses to "
                               "compare. Curated ranges are drawn from the LC "
                               "Classification Outline; uncurated subclasses fall "
                               "back to bucketing by hundreds (e.g., 'F 1400s').")

                    # Default: pick the top 5 subclasses by weight
                    sub_totals = (df_lc.groupby('_lc_sub')['_weight']
                                  .sum()
                                  .sort_values(ascending=False))
                    available_subs = sub_totals.index.tolist()
                    default_subs = available_subs[:5]
                    selected_subs = st.multiselect(
                        "Subclasses to drill into:",
                        options=available_subs,
                        default=default_subs,
                        key="prof_lc_range_subs",
                        help="By default, the top 5 subclasses by weight are shown. "
                             "Pick fewer for a focused view, or pick the specific "
                             "subclasses you're investigating.",
                    )

                    if selected_subs:
                        df_subs = df_lc[df_lc['_lc_sub'].isin(selected_subs)]
                        range_summary = (df_subs.groupby(['_lc_sub', '_lc_range'])
                                         .agg(Records=('_weight', 'count'),
                                              Total=('_weight', 'sum'))
                                         .reset_index()
                                         .rename(columns={
                                             '_lc_sub': 'Subclass',
                                             '_lc_range': 'Range',
                                             'Records': 'Records',
                                             'Total': f'Total {wl}',
                                         }))
                        range_summary = range_summary.sort_values(
                            f'Total {wl}', ascending=False
                        )

                        # Bar chart — combined subclass+range labels for readability
                        # when multiple subclasses are shown together
                        if len(selected_subs) > 1:
                            range_summary['Display'] = (
                                range_summary['Subclass'] + ' — ' + range_summary['Range']
                            )
                            y_col = 'Display'
                        else:
                            y_col = 'Range'

                        # Cap to top 25 ranges by weight to keep the chart readable
                        chart_data = range_summary.head(25)
                        fig_range = px.bar(
                            chart_data,
                            x=f'Total {wl}', y=y_col, orientation='h',
                            color='Subclass' if len(selected_subs) > 1 else None,
                            color_discrete_sequence=[
                                '#285C4D', '#71C5E8', '#92ad9c', '#a8d8e8', '#4a7866',
                            ],
                            title=(f"Range distribution within "
                                   f"{', '.join(selected_subs)} "
                                   f"({len(df_subs):,} records)"),
                            hover_data=['Records'],
                        )
                        fig_range.update_layout(
                            height=max(400, len(chart_data) * 28),
                            yaxis={'categoryorder': 'total ascending'},
                            margin=dict(l=10, r=10, t=60, b=10),
                        )
                        st.plotly_chart(fig_range, use_container_width=True)

                        # Detail table (full, not just top 25)
                        display_df = range_summary[
                            ['Subclass', 'Range', 'Records', f'Total {wl}']
                        ].copy()
                        if has_usage:
                            display_df['% of selected subclass'] = (
                                display_df.groupby('Subclass')[f'Total {wl}']
                                .transform(lambda x: x / x.sum() * 100)
                                .round(1)
                                .astype(str) + '%'
                            )
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                        # Download
                        _range_bytes = _annotate_csv(
                            display_df, notes,
                            extra_meta={'Tool': 'Collection Profiler',
                                        'View': 'Sub-class Range Distribution',
                                        'Weighting': wl,
                                        'Subclasses': ', '.join(selected_subs)}
                        )
                        st.download_button(
                            "📥 Range distribution (CSV)",
                            _range_bytes, "lc_range_distribution.csv",
                            "text/csv", key='prof_dl_ranges',
                        )
                        _add_to_tray("profiler", "lc_range_distribution.csv",
                                     _range_bytes)

                        # --- Range-level Coverage vs. Use ---
                        # Only meaningful when usage data is present. Compares
                        # holdings_share vs. usage_share at the range level
                        # within the selected subclasses, mirroring the
                        # subclass-level CVU but at finer granularity.
                        if has_usage and '_weight' in df_subs.columns:
                            st.markdown("---")
                            st.subheader("Range-level coverage vs. use")
                            st.caption(
                                "Compares **% of selected-subclass holdings** in each range "
                                "against **% of use** that range drives. Same signal logic "
                                "as the LC Main Class view above, but drilled down to LC "
                                "ranges within the subclasses you picked. Uses the same "
                                "thresholds set in the sidebar."
                            )

                            # Build titles_dict and usage_dict at the range level.
                            # Titles count = 1 per row (records); usage = _weight sum.
                            # The reference total is the selected-subclass subtotal so
                            # percentages compare ranges *within* what's been selected,
                            # not the whole collection — that's the more interpretable
                            # question at this granularity.
                            range_titles = (df_subs.groupby('_lc_range')
                                            .size().to_dict())
                            range_usage = (df_subs.groupby('_lc_range')['_weight']
                                           .sum().to_dict())
                            total_t = sum(range_titles.values())
                            total_u = sum(range_usage.values())

                            if total_t > 0 and total_u > 0:
                                # Range labels are self-describing — no need for a
                                # separate label_lookup; pass an empty dict and let
                                # Description show '—'. Then suppress that column.
                                cvu_range_df = _build_cvu_table(
                                    range_titles, range_usage,
                                    total_t, total_u,
                                    settings.get('cvu_over', 2.0),
                                    settings.get('cvu_under', 0.5),
                                    settings.get('cvu_min_titles', 10),
                                    {},  # no lookup — range IS the label
                                    'Range',
                                )
                                if not cvu_range_df.empty:
                                    # Drop Description (always — for range view); sort by signal
                                    cvu_range_df = cvu_range_df.drop(columns=['Description'])
                                    cvu_range_df['_sort_key'] = cvu_range_df[
                                        'Use/Holdings Signal'
                                    ].fillna(-1)
                                    cvu_range_df = (cvu_range_df
                                                    .sort_values('_sort_key', ascending=False)
                                                    .drop(columns='_sort_key'))

                                    # KPI row
                                    k1, k2, k3 = st.columns(3)
                                    over_n = (cvu_range_df['Assessment']
                                              == "🟢 Overperforming").sum()
                                    under_n = (cvu_range_df['Assessment']
                                               == "🔴 Underperforming").sum()
                                    prop_n = (cvu_range_df['Assessment']
                                              == "✅ Proportional").sum()
                                    k1.metric("🟢 Overperforming ranges", int(over_n))
                                    k2.metric("🔴 Underperforming ranges", int(under_n))
                                    k3.metric("✅ Proportional ranges", int(prop_n))

                                    st.dataframe(
                                        cvu_range_df,
                                        use_container_width=True, hide_index=True,
                                        height=400,
                                        column_config={
                                            '% of Collection': st.column_config.NumberColumn(
                                                format="%.2f%%",
                                                help="Share of holdings within the "
                                                     "selected subclass(es). The total "
                                                     "is the selected subset, not the "
                                                     "whole collection.",
                                            ),
                                            '% of Use': st.column_config.NumberColumn(
                                                format="%.2f%%",
                                            ),
                                            'Use/Holdings Signal': st.column_config.NumberColumn(
                                                format="%.2f",
                                                help="(% of use) ÷ (% of holdings). "
                                                     "1.0 = proportional.",
                                            ),
                                        },
                                    )

                                    # Scatter plot
                                    plot_df = cvu_range_df[
                                        cvu_range_df['Assessment'] != "—"
                                    ].copy()
                                    if not plot_df.empty:
                                        fig_cvu_r = px.scatter(
                                            plot_df,
                                            x='% of Collection', y='% of Use',
                                            size='Titles Held', color='Assessment',
                                            color_discrete_map={
                                                "🟢 Overperforming": "#2ecc71",
                                                "🔴 Underperforming": "#e74c3c",
                                                "✅ Proportional": "#71C5E8",
                                            },
                                            hover_data=['Range', 'Total Use',
                                                        'Use/Title Ratio'],
                                            title="Holdings share vs. usage share by range",
                                        )
                                        # Add diagonal reference line (proportional)
                                        max_pct = max(plot_df['% of Collection'].max(),
                                                      plot_df['% of Use'].max())
                                        fig_cvu_r.add_shape(
                                            type='line',
                                            x0=0, y0=0, x1=max_pct, y1=max_pct,
                                            line=dict(color='gray', dash='dot', width=1),
                                        )
                                        fig_cvu_r.update_layout(
                                            height=500,
                                            margin=dict(t=60, l=10, r=10, b=10),
                                        )
                                        st.plotly_chart(fig_cvu_r,
                                                        use_container_width=True)

                                    # Narrative summary
                                    if over_n > 0 or under_n > 0:
                                        bullets = []
                                        if over_n > 0:
                                            top_over = (cvu_range_df[
                                                cvu_range_df['Assessment']
                                                == "🟢 Overperforming"
                                            ].head(3)['Range'].tolist())
                                            bullets.append(
                                                f"- **{over_n} overperforming range"
                                                f"{'s' if over_n != 1 else ''}** — "
                                                f"pulling weight above their holdings "
                                                f"share. Examples: "
                                                f"*{', '.join(top_over)}*. Consider "
                                                f"deeper investment."
                                            )
                                        if under_n > 0:
                                            top_under = (cvu_range_df[
                                                cvu_range_df['Assessment']
                                                == "🔴 Underperforming"
                                            ].head(3)['Range'].tolist())
                                            bullets.append(
                                                f"- **{under_n} underperforming range"
                                                f"{'s' if under_n != 1 else ''}** — "
                                                f"well-represented but lightly used. "
                                                f"Examples: *{', '.join(top_under)}*. "
                                                f"Candidates for weeding review or "
                                                f"reassessment. Switch to the **Title "
                                                f"Analysis** tab and filter by these "
                                                f"ranges to see specific low-use titles."
                                            )
                                        st.markdown("\n".join(bullets))

                                    # Download
                                    _cvu_range_bytes = _annotate_csv(
                                        cvu_range_df, notes,
                                        extra_meta={
                                            'Tool': 'Collection Profiler',
                                            'View': 'Range-level Coverage vs. Use',
                                            'Subclasses': ', '.join(selected_subs),
                                            'Over threshold': settings.get('cvu_over', 2.0),
                                            'Under threshold': settings.get('cvu_under', 0.5),
                                            'Min titles threshold': settings.get('cvu_min_titles', 10),
                                        }
                                    )
                                    st.download_button(
                                        "📥 Range-level CVU (CSV)",
                                        _cvu_range_bytes,
                                        "coverage_vs_use_ranges.csv",
                                        "text/csv", key='prof_dl_cvu_ranges',
                                    )
                                    _add_to_tray("profiler",
                                                 "coverage_vs_use_ranges.csv",
                                                 _cvu_range_bytes)

                                    # --- Drill-down: records behind a chosen range ---
                                    if '_lc_range' in df_subs.columns:
                                        # Order flagged ranges first for quick auditing
                                        _rank_map = {"🟢 Overperforming": 0,
                                                     "🔴 Underperforming": 1,
                                                     "✅ Proportional": 2, "—": 3}
                                        _cvu_sorted = cvu_range_df.copy()
                                        _cvu_sorted['_rank'] = _cvu_sorted[
                                            'Assessment'].map(_rank_map).fillna(4)
                                        range_opts = (_cvu_sorted
                                                      .sort_values('_rank')
                                                      ['Range'].dropna().tolist())
                                        if range_opts:
                                            pick_rng = st.selectbox(
                                                "🔎 Inspect the records behind a range",
                                                options=["(choose a range)"] + range_opts,
                                                key="prof_cvu_range_drill_pick",
                                                help="Flagged ranges are listed first. "
                                                     "Pick one to audit the titles behind "
                                                     "the over/underperforming signal.",
                                            )
                                            if pick_rng != "(choose a range)":
                                                # Pull scope from the records-ctx
                                                # frame (it carries _year); fall
                                                # back to df_subs if ctx absent.
                                                _scope_src = (_records_ctx['df']
                                                              if _records_ctx is not None
                                                              else df_subs)
                                                scope = _scope_src[
                                                    _scope_src['_lc_range'] == pick_rng
                                                ]
                                                _arow = cvu_range_df[
                                                    cvu_range_df['Range'] == pick_rng]
                                                _assess = (_arow['Assessment'].iloc[0]
                                                           if not _arow.empty else "")
                                                _records_drilldown(
                                                    scope,
                                                    key_prefix=f"cvu_rng_{abs(hash(pick_rng)) % 100000}",
                                                    title_col=title_col,
                                                    weight_col=weight_col,
                                                    author_col=author_col,
                                                    location_col=location_col,
                                                    has_year=(_records_ctx or {}).get('has_year', False),
                                                    has_usage=has_usage,
                                                    weight_label=usage_col_label if has_usage else wl,
                                                    notes=notes,
                                                    context_label=f"{pick_rng} {_assess}".strip(),
                                                    expanded=True,
                                                )
                            else:
                                st.info("Need both holdings and usage data within the "
                                        "selected subclasses to compute coverage vs. use.")
                    else:
                        st.info("Pick at least one subclass to see its range distribution.")

    # ======================================================================
    # TAB 2: SUBJECT TERM ANALYSIS
    # ======================================================================
    with tab_subj:
        if not results.get('subject_counts'):
            st.info("No subject data found. Make sure the Subjects column is "
                    "mapped correctly, or upload a file with subject headings.")
        else:
            # --- Subject word cloud ---
            if settings['show_wordcloud'] and results['subject_counts']:
                st.subheader(f"Subject word cloud ({wl}-weighted)")
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
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                                    facecolor='white', edgecolor='none')
                        buf.seek(0)
                        _wc_png_bytes = buf.getvalue()
                        st.download_button("📥 Word cloud (PNG)", _wc_png_bytes,
                                           "collection_wordcloud.png", "image/png",
                                           key='prof_dl_wc')
                        _add_to_tray("profiler", "collection_wordcloud.png", _wc_png_bytes)
                        plt.close(fig)
                    else:
                        st.info(f"No subjects meet the minimum length of {min_len} characters.")

            # --- Top subject bars ---
            if settings['show_subject_bars'] and results['subject_counts']:
                st.markdown("---")
                st.subheader(f"Top {top_n} subject terms")
                top_subjects = results['subject_counts'].most_common(top_n)
                sdf = pd.DataFrame(top_subjects, columns=['Subject', 'Count'])
                fig = px.bar(sdf, x='Count', y='Subject', orientation='h', color='Count',
                             color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  height=max(450, top_n * 24), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                _subj_bytes = _annotate_csv(sdf, notes,
                                            extra_meta={'Tool': 'Collection Profiler',
                                                        'View': 'Top Subjects',
                                                        'Weighting': wl})
                st.download_button("📥 Subject frequencies (CSV)",
                                   _subj_bytes, "subject_frequencies.csv",
                                   "text/csv", key='prof_dl_subj')
                _add_to_tray("profiler", "subject_frequencies.csv", _subj_bytes)

                # --- Drill-down: records tagged with a top subject ---
                if (_records_ctx is not None and _records_ctx.get('subject_col')
                        and _records_ctx['subject_col'] in _records_ctx['df'].columns):
                    subj_bar_opts = [s for s, _ in top_subjects]
                    pick_sb = st.selectbox(
                        "🔎 Inspect the records tagged with a subject",
                        options=["(choose a subject)"] + subj_bar_opts,
                        key="prof_subjbar_drill_pick",
                        help="Records whose subject field contains the chosen term.",
                    )
                    if pick_sb != "(choose a subject)":
                        _sc = _records_ctx['subject_col']
                        _mask = _records_ctx['df'][_sc].astype(str).str.contains(
                            re.escape(pick_sb), case=False, na=False
                        )
                        _records_drilldown(
                            _records_ctx['df'][_mask],
                            key_prefix=f"subjbar_{abs(hash(pick_sb)) % 100000}",
                            title_col=_records_ctx.get('title_col'),
                            weight_col=_records_ctx.get('weight_col'),
                            author_col=_records_ctx.get('author_col'),
                            location_col=_records_ctx.get('location_col'),
                            has_year=_records_ctx.get('has_year', False),
                            has_usage=_records_ctx.get('has_usage', False),
                            weight_label=_records_ctx.get('weight_label', 'Usage'),
                            notes=notes,
                            context_label=f"Subject: {pick_sb}",
                            expanded=True,
                        )

            # --- Title keywords (n-gram analysis) ---
            if settings.get('show_title_keywords'):
                st.markdown("---")
                _render_title_keywords(results, settings, notes=notes)

            # --- Coverage vs. Use (subject-based fallback when LC isn't available) ---
            if settings.get('show_coverage_vs_use') and results.get('cvu_by_subject_available') \
               and not results.get('cvu_available'):
                st.markdown("---")
                _render_coverage_vs_use_by_subject(results, settings, notes=notes,
                                                   records_ctx=_records_ctx)

    # ======================================================================
    # TAB 3: TITLE ANALYSIS (absorbs former Print/Usage Analyzer functionality)
    # ======================================================================
    with tab_title:
        if not title_col:
            st.info("No Title column detected. Title-level analysis is unavailable. "
                    "Map a Title column above to use this tab.")
        elif df is None:
            st.info("Title-level data not available for this analysis pass.")
        else:
            # Filter df to the LC-selected subset (matches what the other tabs see)
            df_view = df.loc[idx] if idx is not None else df

            # ---- Time period filter (year multiselect + optional date range) ----
            # If a date column is detected, build a year-level filter (always) and
            # an additional date-range filter (when the column has sub-year granularity).
            # The year filter is the primary control because most trend questions are
            # year-over-year. Date-range stays available for files with full dates.
            period_label = "unknown period"
            years_in_data = []     # sorted unique years present
            years_selected = []    # years the user has actively included
            has_subyear_granularity = False
            if date_col and date_col in df_view.columns:
                df_view = df_view.copy()
                src = df_view[date_col]
                if pd.api.types.is_numeric_dtype(src):
                    # Year column path: coerce to Jan-1 of that year. No sub-year detail.
                    df_view['_date'] = pd.to_datetime(
                        src.astype('Int64').astype(str) + '-01-01',
                        errors='coerce',
                    )
                else:
                    df_view['_date'] = pd.to_datetime(src, errors='coerce')
                    # Detect sub-year granularity — if any date isn't Jan 1, we have
                    # at least month-level detail and the date-range filter is useful.
                    parsed = df_view['_date'].dropna()
                    if len(parsed) > 0:
                        has_subyear_granularity = bool(
                            ((parsed.dt.month != 1) | (parsed.dt.day != 1)).any()
                        )

                valid_dates = df_view['_date'].dropna()
                if len(valid_dates) > 0:
                    df_view['_year'] = df_view['_date'].dt.year
                    years_in_data = sorted(int(y) for y in df_view['_year'].dropna().unique())
                    period_label = (f"{years_in_data[0]}–{years_in_data[-1]}"
                                    if len(years_in_data) > 1
                                    else f"{years_in_data[0]}")

                    with st.expander(
                        f"📅 Time period filter "
                        f"({len(years_in_data)} year{'s' if len(years_in_data) != 1 else ''} "
                        f"in data: {period_label})",
                        expanded=False,
                    ):
                        # Year multiselect — primary control
                        years_selected = st.multiselect(
                            "Include years:",
                            options=years_in_data,
                            default=years_in_data,
                            key="prof_title_years",
                            help="Pick one or more years to scope all Title Analysis "
                                 "sub-tabs. Defaults to all years; the Yearly trends "
                                 "sub-tab respects this selection too.",
                        )
                        if years_selected and set(years_selected) != set(years_in_data):
                            df_view = df_view[df_view['_year'].isin(years_selected)].copy()
                            if len(years_selected) > 1:
                                period_label = (
                                    f"{min(years_selected)}–{max(years_selected)}"
                                )
                            else:
                                period_label = f"{years_selected[0]}"
                            st.caption(f"Filtered to {len(df_view):,} records "
                                       f"in {period_label}")

                        # Date-range slider — only for sub-year granularity, secondary control
                        if has_subyear_granularity and len(valid_dates) > 0:
                            use_dr = st.checkbox(
                                "Also apply a date range within the selected years",
                                value=False, key="prof_title_use_dr",
                            )
                            if use_dr:
                                # Recompute min/max after year filter
                                vd = df_view['_date'].dropna()
                                if len(vd) > 0:
                                    dmin_d = vd.min().date()
                                    dmax_d = vd.max().date()
                                    start_date, end_date = st.date_input(
                                        "Date range:",
                                        value=(dmin_d, dmax_d),
                                        min_value=dmin_d, max_value=dmax_d,
                                        key="prof_title_date_range",
                                    )
                                    if isinstance(start_date, tuple):
                                        start_date, end_date = start_date
                                    mask = (df_view['_date'].dt.date >= start_date) & \
                                           (df_view['_date'].dt.date <= end_date)
                                    df_view = df_view[mask].copy()
                                    period_label = _format_date_range(
                                        pd.Timestamp(start_date),
                                        pd.Timestamp(end_date),
                                    )

            # If no year selection happened above, treat all years as selected
            if not years_selected and years_in_data:
                years_selected = years_in_data

            # ---- Sub-tabs within Title Analysis ----
            if has_usage and weight_col and '_weight' in df_view.columns:
                # Yearly trends appears first when we have >1 year of data —
                # multi-year files want this view as the headline.
                title_subtabs = []
                show_yearly = len(years_in_data) > 1 and '_year' in df_view.columns
                if show_yearly:
                    title_subtabs.append("Yearly trends")
                title_subtabs.extend(["Top titles", "Weeding review"])
                if author_col:
                    title_subtabs.append("Author summary")
                title_subtabs.append("Title details")
                ts_objs = st.tabs(title_subtabs)
                ts_idx = {label: i for i, label in enumerate(title_subtabs)}

                # ---- Yearly trends (only when >1 year of data) ----
                if show_yearly:
                    with ts_objs[ts_idx["Yearly trends"]]:
                        st.info("How usage shifts year-over-year. Useful for "
                                "spotting growing or declining areas across multi-year datasets.")

                        # Per-year aggregates
                        year_agg = df_view.groupby('_year').agg(
                            Records=('_weight', 'count'),
                            Total=('_weight', 'sum'),
                            Mean=('_weight', 'mean'),
                            Median=('_weight', 'median'),
                        ).reset_index().rename(columns={
                            '_year': 'Year',
                            'Records': 'Records',
                            'Total': f'Total {weight_col}',
                            'Mean': f'Mean {weight_col}',
                            'Median': f'Median {weight_col}',
                        })
                        # Force Year as int (no decimals in display)
                        year_agg['Year'] = year_agg['Year'].astype(int)

                        # KPI row
                        kc1, kc2, kc3 = st.columns(3)
                        kc1.metric("Years in selection", f"{len(year_agg)}")
                        kc2.metric(f"Total {weight_col}",
                                   f"{int(year_agg[f'Total {weight_col}'].sum()):,}")
                        avg_per_year = year_agg[f'Total {weight_col}'].mean() if len(year_agg) else 0
                        kc3.metric(f"Avg {weight_col} per year", f"{int(round(avg_per_year)):,}")

                        # Trend bar chart
                        fig_yr = px.bar(
                            year_agg, x='Year', y=f'Total {weight_col}',
                            title=f"{weight_col} by year — {period_label}",
                            color=f'Total {weight_col}',
                            color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                            hover_data=['Records', f'Mean {weight_col}', f'Median {weight_col}'],
                        )
                        fig_yr.update_layout(height=400, showlegend=False,
                                             xaxis=dict(type='category'))
                        st.plotly_chart(fig_yr, use_container_width=True)
                        st.dataframe(year_agg, use_container_width=True, hide_index=True)

                        # Year totals download
                        _yr_bytes = _annotate_csv(
                            year_agg, notes,
                            extra_meta={'Tool': 'Collection Profiler',
                                        'View': 'Yearly trends',
                                        'Metric': weight_col,
                                        'Period': period_label}
                        )
                        st.download_button(
                            "📥 Yearly totals (CSV)",
                            _yr_bytes, f"yearly_totals_{_slug_period(period_label)}.csv".replace('_.', '.'),
                            "text/csv", key="prof_title_dl_year",
                        )
                        _add_to_tray("profiler",
                                     f"yearly_totals_{_slug_period(period_label)}.csv".replace('_.', '.'),
                                     _yr_bytes)

                        # Top-N titles year-over-year
                        st.markdown("---")
                        st.markdown("**Top titles across years**")
                        n_yoy = st.slider(
                            "How many top titles to track?",
                            3, 25, 10, key="prof_title_yoy_n",
                            help="Picks the top N titles overall (by total usage across "
                                 "selected years), then shows their usage in each year.",
                        )
                        # Identify overall top N titles
                        top_titles_overall = (df_view.groupby(title_col)['_weight']
                                              .sum()
                                              .nlargest(n_yoy)
                                              .index.tolist())
                        if top_titles_overall:
                            yoy_subset = df_view[df_view[title_col].isin(top_titles_overall)]
                            yoy_pivot = (yoy_subset
                                         .groupby([title_col, '_year'])['_weight']
                                         .sum()
                                         .reset_index()
                                         .rename(columns={'_weight': weight_col,
                                                          '_year': 'Year'}))
                            yoy_pivot['Year'] = yoy_pivot['Year'].astype(int)

                            fig_yoy = px.line(
                                yoy_pivot, x='Year', y=weight_col, color=title_col,
                                title=f"Top {n_yoy} titles — usage trajectory",
                                markers=True,
                            )
                            fig_yoy.update_layout(
                                height=max(400, n_yoy * 20),
                                xaxis=dict(type='category'),
                            )
                            st.plotly_chart(fig_yoy, use_container_width=True)

                            # Wide-format table: titles × years
                            yoy_wide = yoy_pivot.pivot(
                                index=title_col, columns='Year', values=weight_col
                            ).fillna(0).astype(int)
                            yoy_wide = yoy_wide.reindex(top_titles_overall)
                            yoy_wide['Total'] = yoy_wide.sum(axis=1)
                            yoy_wide = yoy_wide.sort_values('Total', ascending=False)
                            st.dataframe(yoy_wide, use_container_width=True)

                            # Download
                            _yoy_bytes = _annotate_csv(
                                yoy_wide.reset_index(), notes,
                                extra_meta={'Tool': 'Collection Profiler',
                                            'View': 'Yearly trends — top titles',
                                            'Metric': weight_col,
                                            'Top N': str(n_yoy),
                                            'Period': period_label}
                            )
                            st.download_button(
                                "📥 Top titles by year (CSV)",
                                _yoy_bytes,
                                f"top_titles_by_year_{_slug_period(period_label)}.csv".replace('_.', '.'),
                                "text/csv", key="prof_title_dl_yoy",
                            )
                            _add_to_tray("profiler",
                                         f"top_titles_by_year_{_slug_period(period_label)}.csv".replace('_.', '.'),
                                         _yoy_bytes)

                # ---- Top titles ----
                with ts_objs[ts_idx["Top titles"]]:
                    n_top = st.slider("Show Top N Titles", 5, 100, 25, key="prof_title_topn")
                    cols_show = [title_col, '_weight']
                    if author_col and author_col in df_view.columns:
                        cols_show.insert(1, author_col)
                    if location_col and location_col in df_view.columns:
                        cols_show.append(location_col)
                    top_titles = df_view.nlargest(n_top, '_weight')[cols_show].rename(
                        columns={'_weight': weight_col}
                    )
                    fig_top = px.bar(
                        top_titles, x=weight_col, y=title_col, orientation='h',
                        title=f"Top {n_top} Titles by {weight_col} — {period_label}",
                        color=weight_col,
                        color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                    )
                    fig_top.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=max(450, n_top * 22),
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
                    st.dataframe(top_titles, use_container_width=True, hide_index=True)

                    _top_bytes = _annotate_csv(
                        top_titles, notes,
                        extra_meta={'Tool': 'Collection Profiler',
                                    'View': 'Top Titles',
                                    'Metric': weight_col,
                                    'Period': period_label}
                    )
                    _top_fname = f"top_titles_{_slug_period(period_label)}.csv".replace('_.', '.')
                    st.download_button("📥 Top titles (CSV)",
                                       _top_bytes, _top_fname, "text/csv",
                                       key="prof_title_dl_top")
                    _add_to_tray("profiler", _top_fname, _top_bytes)

                # ---- Weeding review (low-usage threshold) ----
                with ts_objs[ts_idx["Weeding review"]]:
                    st.info("Review titles with low or zero usage for potential "
                            "weeding, off-site storage, or replacement.")
                    max_w = int(df_view['_weight'].max()) if len(df_view) else 100
                    threshold = st.number_input(
                        f"Low-Usage Threshold ({weight_col})",
                        min_value=0, max_value=max_w, value=0,
                        key="prof_title_thr",
                    )
                    cand = df_view[df_view['_weight'] <= threshold].copy()

                    sort_choice = st.radio(
                        "Sort candidates by",
                        ["Lowest use first", "Call number (shelf order)"],
                        horizontal=True, key="prof_title_weed_sort",
                        help="Shelf order sorts by LC subclass then number, so the "
                             "list doubles as a shelf-ready pull sheet.")
                    if (sort_choice.startswith("Call number")
                            and '_lc_sub' in cand.columns):
                        cand = cand.sort_values(['_lc_sub', '_lc_number', '_weight'],
                                                na_position='last')
                    else:
                        cand = cand.sort_values('_weight')

                    low_cols = [title_col, '_weight']
                    if author_col and author_col in cand.columns:
                        low_cols.insert(1, author_col)
                    if lc_col and lc_col in cand.columns:
                        low_cols.append(lc_col)            # full call number (shelf-ready)
                    if '_lc_main' in cand.columns:
                        low_cols.append('_lc_main')
                    if location_col and location_col in cand.columns:
                        low_cols.append(location_col)
                    rename_map = {'_weight': weight_col, '_lc_main': 'LC Class'}
                    if lc_col:
                        rename_map[lc_col] = 'Call Number'
                    low_use = cand[low_cols].rename(columns=rename_map)

                    cc1, cc2 = st.columns(2)
                    cc1.metric("Titles ≤ Threshold", f"{len(low_use):,}")
                    cc2.metric("% of Collection",
                               f"{len(low_use) / max(1, len(df_view)) * 100:.1f}%")
                    st.dataframe(low_use, use_container_width=True, height=400)

                    # ---- Where the weeding candidates cluster (subject + LC) ----
                    if len(low_use):
                        st.markdown("---")
                        st.markdown("**Where the weeding candidates cluster**")
                        tcol1, tcol2 = st.columns(2)
                        with tcol1:
                            # Finer granularity: break candidates by LC sub-class
                            # range (e.g., HQ1101–2030.7) when available; fall back
                            # to the two-letter class.
                            if '_lc_range' in cand.columns and cand['_lc_range'].notna().any():
                                rc = cand['_lc_range'].dropna().value_counts().head(15)
                                rdf = pd.DataFrame({'Sub-class range': rc.index,
                                                    'Candidates': rc.values})
                                figL = px.bar(rdf, x='Candidates', y='Sub-class range',
                                              orientation='h', color='Candidates',
                                              color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                                              title="Candidates by LC sub-class range")
                                figL.update_layout(yaxis={'categoryorder': 'total ascending'},
                                                   height=max(300, len(rdf) * 26),
                                                   showlegend=False, margin=dict(l=4, r=4, t=40, b=4))
                                st.plotly_chart(figL, use_container_width=True)
                            elif '_lc_main' in cand.columns and cand['_lc_main'].notna().any():
                                lc_counts = cand['_lc_main'].dropna().value_counts().head(15)
                                lcdf = pd.DataFrame({
                                    'LC Class': [f"{c} – {LC_CLASSES.get(c, '?')}"
                                                 for c in lc_counts.index],
                                    'Candidates': lc_counts.values,
                                })
                                figL = px.bar(lcdf, x='Candidates', y='LC Class',
                                              orientation='h', color='Candidates',
                                              color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                                              title="Candidates by LC class")
                                figL.update_layout(yaxis={'categoryorder': 'total ascending'},
                                                   height=max(300, len(lcdf) * 26),
                                                   showlegend=False, margin=dict(l=4, r=4, t=40, b=4))
                                st.plotly_chart(figL, use_container_width=True)
                            else:
                                st.caption("No LC data for these candidates.")
                        with tcol2:
                            cand_subj = Counter()
                            if subj_col and subj_col in cand.columns and cand[subj_col].notna().any():
                                _profiler_process_subjects_chunk(
                                    cand[subj_col], pd.Series(1.0, index=cand.index),
                                    pd.Series(None, index=cand.index),
                                    cand_subj, defaultdict(Counter))
                            top_cs = cand_subj.most_common(15)
                            if top_cs:
                                csdf = pd.DataFrame(top_cs, columns=['Subject', 'Candidates'])
                                figS = px.bar(csdf, x='Candidates', y='Subject',
                                              orientation='h', color='Candidates',
                                              color_continuous_scale=[[0, '#71C5E8'], [1, '#285C4D']],
                                              title="Candidates by subject term")
                                figS.update_layout(yaxis={'categoryorder': 'total ascending'},
                                                   height=max(300, len(csdf) * 26),
                                                   showlegend=False, margin=dict(l=4, r=4, t=40, b=4))
                                st.plotly_chart(figS, use_container_width=True)
                            else:
                                st.caption("No subject data for these candidates "
                                           "(map a Subjects column to enable this).")

                        # Concentration summary
                        bits = []
                        if '_lc_range' in cand.columns and cand['_lc_range'].notna().any():
                            rr = cand['_lc_range'].dropna().value_counts()
                            bits.append(f"the **{rr.index[0]}** range "
                                        f"({int(rr.iloc[0]):,} titles)")
                        elif '_lc_main' in cand.columns and cand['_lc_main'].notna().any():
                            tl = cand['_lc_main'].dropna().value_counts()
                            bits.append(f"the **{tl.index[0]} – {LC_CLASSES.get(tl.index[0], '?')}** "
                                        f"class ({int(tl.iloc[0]):,} titles, "
                                        f"{tl.iloc[0] / max(1, len(cand)) * 100:.0f}%)")
                        if top_cs:
                            bits.append(f"the subject **{top_cs[0][0]}** "
                                        f"({int(top_cs[0][1]):,} titles)")
                        if bits:
                            st.caption("Weeding candidates concentrate most in "
                                       + " and ".join(bits) + ".")

                        # Downloadable breakdowns for these candidates
                        breakdowns = []
                        if '_lc_range' in cand.columns and cand['_lc_range'].notna().any():
                            rng_tab = (cand['_lc_range'].dropna().value_counts()
                                       .rename_axis('Sub-class range').reset_index(name='Candidates'))
                            breakdowns.append(("by sub-class range", rng_tab))
                        if '_lc_main' in cand.columns and cand['_lc_main'].notna().any():
                            lc_tab = (cand['_lc_main'].dropna().value_counts()
                                      .rename_axis('LC Class').reset_index(name='Candidates'))
                            lc_tab['LC Class'] = lc_tab['LC Class'].map(
                                lambda c: f"{c} – {LC_CLASSES.get(c, '?')}")
                            breakdowns.append(("by LC class", lc_tab))
                        if top_cs:
                            breakdowns.append(("by subject",
                                               pd.DataFrame(cand_subj.most_common(),
                                                            columns=['Subject', 'Candidates'])))
                        for blabel, btab in breakdowns:
                            _bd_bytes = _annotate_csv(
                                btab, notes,
                                extra_meta={'Tool': 'Use Analysis', 'View': f'Weeding candidates {blabel}',
                                            'Metric': weight_col, 'Threshold': threshold,
                                            'Period': period_label})
                            st.download_button(
                                f"📥 Candidate breakdown {blabel} (CSV)", _bd_bytes,
                                f"weeding_candidates_{blabel.replace(' ', '_')}.csv", "text/csv",
                                key=f"prof_weed_bd_{blabel.replace(' ', '_')}")

                    _weed_fname = f"weeding_review_{_slug_period(period_label)}.csv".replace('_.', '.')
                    _weed_bytes = _annotate_csv(
                        low_use, notes,
                        extra_meta={'Tool': 'Collection Profiler',
                                    'View': 'Weeding Review',
                                    'Metric': weight_col,
                                    'Threshold': threshold,
                                    'Period': period_label}
                    )
                    st.download_button("📥 Weeding review list (CSV)",
                                       _weed_bytes, _weed_fname, "text/csv",
                                       key="prof_title_dl_weed")
                    _add_to_tray("profiler", _weed_fname, _weed_bytes)

                # ---- Author summary ----
                if author_col and "Author summary" in ts_idx:
                    with ts_objs[ts_idx["Author summary"]]:
                        if author_col in df_view.columns:
                            auth_summary = df_view.groupby(author_col).agg(
                                **{
                                    'Title Count': (title_col, 'count'),
                                    f'Total {weight_col}': ('_weight', 'sum'),
                                }
                            ).reset_index().sort_values(f'Total {weight_col}', ascending=False).head(100)
                            st.markdown(f"**Top 100 authors by total {weight_col}:**")
                            st.dataframe(auth_summary, use_container_width=True,
                                         hide_index=True, height=500)
                            _auth_fname = f"author_summary_{_slug_period(period_label)}.csv".replace('_.', '.')
                            _auth_bytes = _annotate_csv(
                                auth_summary, notes,
                                extra_meta={'Tool': 'Collection Profiler',
                                            'View': 'Author Summary',
                                            'Metric': weight_col,
                                            'Period': period_label}
                            )
                            st.download_button("📥 Author summary (CSV)",
                                               _auth_bytes, _auth_fname, "text/csv",
                                               key="prof_title_dl_auth")
                            _add_to_tray("profiler", _auth_fname, _auth_bytes)

                # ---- Title details (paginated) ----
                with ts_objs[ts_idx["Title details"]]:
                    if results.get('detail_available'):
                        PAGE_SIZE = 5_000
                        total = len(df_view)
                        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
                        page = st.number_input("Page", 1, total_pages, 1, key='prof_page')
                        start = (page - 1) * PAGE_SIZE
                        end = min(start + PAGE_SIZE, total)
                        st.caption(f"Records {start + 1:,}–{end:,} of {total:,}")
                        detail_cols = [c for c in [title_col, author_col, '_lc_main',
                                                   results.get('detail_cols', [None, None, None])[-1]
                                                   if results.get('detail_cols') else None]
                                       if c and c in df_view.columns]
                        if '_weight' in df_view.columns:
                            detail_cols.append('_weight')
                        view_slice = df_view.iloc[start:end][detail_cols].rename(
                            columns={'_weight': weight_col, '_lc_main': 'LC Class'}
                        )
                        st.dataframe(view_slice, use_container_width=True,
                                     height=400, hide_index=True)
                    else:
                        st.info("Title details unavailable for this file.")
            else:
                # No usage column — only show title details
                st.info("No usage column detected. The Top Titles, Weeding Review, "
                        "and Author Summary views are usage-dependent and only "
                        "appear when a circulation/views/loans column is present. "
                        "You can still browse title details below.")
                if results.get('detail_available'):
                    PAGE_SIZE = 5_000
                    total = results.get('detail_total', len(df_view))
                    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
                    page = st.number_input("Page", 1, total_pages, 1, key='prof_page')
                    start = (page - 1) * PAGE_SIZE
                    end = min(start + PAGE_SIZE, total)
                    st.caption(f"Records {start + 1:,}–{end:,} of {total:,}")
                    detail_cols = results.get('detail_cols', [])
                    if detail_cols:
                        page_idx = idx[start:end] if idx is not None else df.index[start:end]
                        st.dataframe(df.loc[page_idx, detail_cols],
                                     use_container_width=True, height=400, hide_index=True)

    # ---- Consolidated download tray (across all tabs) ----
    st.markdown("---")
    st.subheader("Downloads")
    _render_download_tray("profiler", zip_filename="collection_profiler_results.zip")


def _profiler_ui(mode="structure", flavor="print"):
    """Shared collection-analysis UI, parameterized by mode.

    mode="structure"  -> Collection Profiler page: holdings structure only
        (LC, Subject Term, sub-class ranges, distribution). No usage views.
    mode="usage"      -> Use Analysis tool (print / other-usage branch): the
        same engine with usage on, exposing Coverage-vs-Use, gap, and
        usage-weighted title analysis. Expects the synced explicit-zero
        master from the Zero-Use Identifier as input.

    Widget/session/cache keys carry the `KP` prefix so the two modes keep
    independent state when the user switches between tools.
    """
    KP = "" if mode == "structure" else "use_"

    if f'{KP}prof_results' not in st.session_state:
        st.session_state[f'{KP}prof_results'] = None
    if f'{KP}prof_filtered_idx' not in st.session_state:
        st.session_state[f'{KP}prof_filtered_idx'] = None

    if mode == "structure":
        st.header("\U0001F5FA\uFE0F Collection Profiler")
        st.markdown(
            "**What does our collection look like?** Upload a title list from "
            "Alma or a vendor with subject terms and/or LC call numbers to map "
            "disciplinary strengths and explore subject coverage. This view is "
            "structure-only \u2014 for usage-driven analysis (Coverage vs. Use, "
            "cost-per-use, dead weight) use the **Use Analysis** tool."
        )
        with st.expander("\u2139\uFE0F When to use this tool"):
            st.markdown(
                "- **Collections:** Baseline assessment, accreditation self-studies, "
                "weeding prep (find thin/missing areas), justifying budget asks by "
                "showing strengths.\n"
                "- **Instruction:** Prepare for a liaison session \u2014 see at a glance "
                "what you actually have in HQ or PN before walking into the class.\n"
                "- **Outreach:** Quick visuals for faculty meetings and annual reports "
                "(\"here's what sociology looks like in our collection\")."
            )
        _uploader_label = "\U0001F4C2 Upload title list (CSV or Excel)"
        _uploader_help = ("Needs Subjects and/or LC Classification. Optimized for "
                          "500K\u20131M+ records. Excel (XLS, XLSX) also supported "
                          "\u2014 useful for vendor admin exports.")
        _empty_hint = ("\U0001F4A1 Your file should have some combination of "
                       "**Subjects**, **LC Classification** or **Call Number**, "
                       "and **Title**.")
    else:
        if flavor == "print":
            st.markdown(
                "**Print circulation.** Upload the synced **explicit-zero master** "
                "from the Zero-Use Identifier (Title + Subjects/LC + circulation, with "
                "unused titles carried as 0). You'll get Coverage vs. Use, top titles, "
                "and gap-vs-use across LC and subject."
            )
        else:
            st.markdown(
                "**Other usage data.** Upload a synced title-level file with a usage "
                "column (plus Subjects/LC if you have them) \u2014 ideally the "
                "explicit-zero master from the Zero-Use Identifier so unused titles "
                "count as 0. Coverage vs. Use unlocks when Subjects/LC are present."
            )
        _uploader_label = "\U0001F4C2 Upload synced usage file (CSV or Excel)"
        _uploader_help = ("Title-level file with a usage column. Include Subjects "
                          "and/or LC to unlock Coverage vs. Use. The **All titles "
                          "(explicit zeros)** output from the Zero-Use Identifier is "
                          "the intended input.")
        _empty_hint = ("\U0001F4A1 Feed the **All titles (explicit zeros)** output "
                       "from the Zero-Use Identifier: Title + a usage column, plus "
                       "Subjects/LC to unlock Coverage vs. Use.")

    uploaded_file = st.file_uploader(
        _uploader_label, type=['csv', 'xls', 'xlsx'],
        help=_uploader_help,
        key=f"{KP}prof_upload"
    )

    if uploaded_file is None:
        st.caption(_empty_hint)
        return

    # Friendly error if user uploaded an Excel file but the required library
    # isn't installed. Pandas raises a low-level ImportError otherwise.
    _fname_lower = uploaded_file.name.lower()
    if _fname_lower.endswith('.xlsx') and not XLSX_AVAILABLE:
        st.error("❌ This XLSX file needs the `openpyxl` library. Install with "
                 "`pip install openpyxl`, then re-upload. (Or save the file as "
                 "CSV from Excel and upload that.)")
        return
    if _fname_lower.endswith('.xls') and not XLS_AVAILABLE:
        st.error("❌ This XLS file needs the `xlrd` library (version 2.0.1+). "
                 "Install with `pip install xlrd`, then re-upload. (Or save the "
                 "file as CSV from Excel and upload that.)")
        return

    # Check session cache first — if we've already processed this file in
    # this session, skip the load + LC extraction + weight coercion entirely
    cached_df = _cached_df_for_tool(f"{KP}profiler", uploaded_file)
    if cached_df is not None:
        df = cached_df
        st.success(f"✅ Using cached data for *{uploaded_file.name}* "
                   f"({len(df):,} records)")
        # Re-detect columns from cached df
        subj_col = find_column(df, SUBJECT_ALIASES)
        lc_col = find_column(df, LC_ALIASES)
        title_col = find_column(df, TITLE_ALIASES)
        weight_col = find_column(df, WEIGHT_ALIASES)
        # Derive weight_label the same way the fresh path does
        weight_label = weight_col if weight_col else "Title Count"
    else:
        file_bytes = uploaded_file.getvalue()
        all_cols = _detect_columns_from_header(file_bytes, uploaded_file.name)
        subj_col = find_column(all_cols, SUBJECT_ALIASES)
        lc_col = find_column(all_cols, LC_ALIASES)
        title_col = find_column(all_cols, TITLE_ALIASES)
        weight_col = find_column(all_cols, WEIGHT_ALIASES)

        # Load all columns so the manual-override dropdown has full options.
        # This is a small cost vs. the prior partial-load optimization, but it
        # means users can recover when our aliases miss their column names.
        with st.spinner(f"Loading {uploaded_file.name}..."):
            df = _load_csv_chunked(file_bytes, uploaded_file.name, cols_to_keep=None)

        st.success(f"✅ Loaded **{len(df):,}** records from *{uploaded_file.name}*")

        subj_col = find_column(df, SUBJECT_ALIASES)
        lc_col = find_column(df, LC_ALIASES)
        title_col = find_column(df, TITLE_ALIASES)
        weight_col = find_column(df, WEIGHT_ALIASES)

        if weight_col:
            df['_weight'] = pd.to_numeric(df[weight_col], errors='coerce').fillna(1)
            weight_label = weight_col
        else:
            df['_weight'] = 1.0
            weight_label = "Title Count"

        if lc_col:
            df['_lc_main'], df['_lc_sub'], df['_lc_number'] = _extract_lc_vectorized(df[lc_col])
            # Vectorize range lookup using zip — small loop over rows, but each
            # call is cheap and the total work scales with row count, not records.
            df['_lc_range'] = [
                lookup_lc_range(sub, num)
                for sub, num in zip(df['_lc_sub'], df['_lc_number'])
            ]
        else:
            df['_lc_main'] = None
            df['_lc_sub'] = None
            df['_lc_number'] = None
            df['_lc_range'] = None

        # Save to cache for future tool switches
        _store_cached_df(f"{KP}profiler", uploaded_file, df)

    # Track prior LC column so we re-extract if user changes it
    _original_lc_col = find_column(df, LC_ALIASES)

    # Column detection + manual override
    detected_ok = bool(subj_col or lc_col)
    with st.expander(
        "🔍 Column mapping" + ("" if detected_ok else " — ⚠️ action needed"),
        expanded=not detected_ok
    ):
        if detected_ok:
            st.caption("We auto-detected these columns. Override any of them if we guessed wrong.")
        else:
            st.warning(
                "We couldn't automatically find a Subjects or LC Classification column. "
                "Pick the right ones from your file below — you need at least one of Subjects or LC."
            )
        all_cols = list(df.columns)
        # Hide internal columns we added
        all_cols = [c for c in all_cols if not c.startswith('_')]
        none_opt = "— none —"
        options = [none_opt] + all_cols

        def _idx(col):
            return options.index(col) if col in all_cols else 0

        mc1, mc2 = st.columns(2)
        with mc1:
            subj_pick = st.selectbox("Subjects column", options, index=_idx(subj_col), key=f"{KP}prof_map_subj")
            title_pick = st.selectbox("Title column (optional)", options, index=_idx(title_col), key=f"{KP}prof_map_title")
        with mc2:
            lc_pick = st.selectbox("LC / Call Number column", options, index=_idx(lc_col), key=f"{KP}prof_map_lc")
            weight_pick = st.selectbox("Usage / weight column (optional)", options, index=_idx(weight_col), key=f"{KP}prof_map_weight")

        subj_col = None if subj_pick == none_opt else subj_pick
        lc_col = None if lc_pick == none_opt else lc_pick
        title_col = None if title_pick == none_opt else title_pick
        new_weight_col = None if weight_pick == none_opt else weight_pick

    # If user changed the weight column, re-coerce _weight
    if new_weight_col != weight_col:
        weight_col = new_weight_col
        if weight_col:
            df['_weight'] = pd.to_numeric(df[weight_col], errors='coerce').fillna(1)
            weight_label = weight_col
        else:
            df['_weight'] = 1.0
            weight_label = "Title Count"

    # If user picked a different LC column than what was auto-detected, re-extract
    if lc_col and lc_col != _original_lc_col:
        df['_lc_main'], df['_lc_sub'], df['_lc_number'] = _extract_lc_vectorized(df[lc_col])
        df['_lc_range'] = [
            lookup_lc_range(sub, num)
            for sub, num in zip(df['_lc_sub'], df['_lc_number'])
        ]
    elif not lc_col:
        df['_lc_main'] = None
        df['_lc_sub'] = None
        df['_lc_number'] = None
        df['_lc_range'] = None

    if not subj_col and not lc_col:
        st.error("❌ Please pick at least a Subjects or LC Classification column above to continue.")
        return

    # Detect additional columns the Title Analysis tab uses. Silent detection —
    # these enable optional features but aren't required.
    author_col = find_column(df, AUTHOR_ALIASES)
    location_col = find_column(df, LOCATION_ALIASES)
    date_col = _detect_print_date_column(df)

    # Sidebar: weight mode + LC filter only (upstream choices that change what's analyzed)
    with st.sidebar:
        st.subheader("Analysis settings")
        weight_options = ["Title count (1 per title)"]
        if weight_col:
            weight_options.append(f"Usage metric ({weight_label})")
        if mode == "usage":
            # Usage analysis: default to weighting by the usage column when present
            _w_idx = 1 if (weight_col and len(weight_options) > 1) else 0
            analysis_mode = st.radio("Weight titles by:", weight_options,
                                     index=_w_idx, key=f"{KP}prof_mode")
            use_weight = weight_col and "Usage" in analysis_mode
        else:
            # Structure-only profiling: usage never drives the analysis
            analysis_mode = weight_options[0]
            use_weight = False

        if lc_col:
            st.markdown("---")
            st.subheader("Filter by LC class")
            avail = sorted(df['_lc_main'].dropna().unique())
            labels = [f"{c} – {LC_CLASSES.get(c, '?')}" for c in avail]
            sel_labels = st.multiselect("Include:", labels, default=labels, key=f"{KP}prof_lc_filter")
            sel_classes = [l.split(' –')[0] for l in sel_labels]
            # "Active" = the user has actually narrowed the selection. When active,
            # filtering is strict (unclassified records are excluded too); when all
            # classes are selected, everything passes through including unclassified.
            lc_filter_active = bool(avail) and set(sel_classes) != set(avail)
        else:
            sel_classes = None
            lc_filter_active = False

    # Main-body: visualization toggles in a collapsed "Customize view" expander
    with st.expander("🎨 Customize view (visualizations, word cloud, thresholds)", expanded=False):
        vc1, vc2 = st.columns(2)
        with vc1:
            top_n = st.slider("Top N subjects", 10, 100, 30, 5, key=f"{KP}prof_topn")
            show_sunburst = st.checkbox("LC sunburst", True, key=f"{KP}prof_sun")
            show_treemap = st.checkbox("LC treemap", True, key=f"{KP}prof_tree")
            show_bars = st.checkbox("Top subjects bar chart", True, key=f"{KP}prof_bars")
            show_wordcloud = st.checkbox("Subject word cloud", True, key=f"{KP}prof_wc")
        with vc2:
            show_heatmap = st.checkbox("LC × subject heatmap", True, key=f"{KP}prof_heat")
            show_title_keywords = st.checkbox(
                "Title keywords (uncontrolled vocabulary)",
                value=bool(title_col),
                disabled=not title_col,
                key=f"{KP}prof_tk",
                help="A supplementary lens that tokenizes title text (with stopwords "
                     "stripped) — useful for surfacing terminology that subject "
                     "headings may have missed. Requires a Title column. Distinct "
                     "from the subject view above."
            )
            if mode == "usage":
                show_coverage_vs_use = st.checkbox(
                    "Coverage vs. Use (requires usage column)",
                    value=bool(weight_col),
                    disabled=not weight_col,
                    key=f"{KP}prof_cvu",
                    help="Compare what % of the collection each area holds vs. what % "
                         "of use it drives. Shows overperforming (small but heavily used) "
                         "and underperforming (large but lightly used) areas. "
                         "Uses LC class when available; falls back to subject terms otherwise."
                )
                show_gap = st.checkbox("Gap analysis", True, key=f"{KP}prof_gap")
            else:
                show_coverage_vs_use = False
                show_gap = False
            show_detail = st.checkbox("Title detail table", False, key=f"{KP}prof_detail")

        # Coverage-vs-Use threshold configuration
        if show_coverage_vs_use and weight_col:
            with st.expander("⚖️ Coverage vs. Use thresholds"):
                st.caption("Signal = (% of use) ÷ (% of holdings). A value of 1.0 means "
                           "usage is proportional to holdings.")
                cvu_over = st.slider(
                    "Overperforming threshold (≥)",
                    min_value=1.1, max_value=5.0, value=2.0, step=0.1,
                    key=f"{KP}prof_cvu_over",
                    help="Ratio at or above this flags an area as overperforming "
                         "(higher % of use than of holdings). Lower = more areas flagged."
                )
                cvu_under = st.slider(
                    "Underperforming threshold (≤)",
                    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                    key=f"{KP}prof_cvu_under",
                    help="Ratio at or below this flags an area as underperforming "
                         "(lower % of use than of holdings). Higher = more areas flagged."
                )
                cvu_min_titles = st.number_input(
                    "Minimum titles to include in signal",
                    min_value=1, max_value=1000, value=10,
                    key=f"{KP}prof_cvu_min",
                    help="LC areas with fewer titles than this won't get a signal "
                         "label (too little data to draw conclusions). They still appear "
                         "in the table marked as '—'."
                )
                cvu_show_sub = st.checkbox(
                    "Also break down by LC subclass",
                    value=True,
                    key=f"{KP}prof_cvu_sub",
                    help="Shows e.g. HQ1000s separately from HQ750s."
                )
        else:
            cvu_over, cvu_under, cvu_min_titles, cvu_show_sub = 2.0, 0.5, 10, True

        # Word cloud sub-options (only shown when word cloud is on)
        if show_wordcloud:
            with st.expander("☁️ Word cloud options"):
                wc_max_words = st.slider("Max words", 20, 200, 100, 10, key=f"{KP}prof_wc_max")
                wc_min_len = st.slider("Min word length", 1, 10, 3, key=f"{KP}prof_wc_min")
                wc_color = st.selectbox(
                    "Color scheme",
                    ["viridis", "plasma", "inferno", "magma", "cividis", "twilight", "rainbow"],
                    key=f"{KP}prof_wc_color"
                )
        else:
            wc_max_words, wc_min_len, wc_color = 100, 3, "viridis"

        # Title-keyword sub-options
        if show_title_keywords:
            with st.expander("🔤 Title keyword options"):
                tk_top_n = st.slider(
                    "Top N keywords (per n-gram size)", 10, 100, 30, 5, key=f"{KP}prof_tk_topn",
                    help="Number of keywords/phrases to chart and include in the "
                         "table for each n-gram size you've selected."
                )
                tk_ngram_choice = st.multiselect(
                    "N-gram sizes",
                    options=["Single words (unigrams)",
                             "Two-word phrases (bigrams)",
                             "Three-word phrases (trigrams)"],
                    default=["Single words (unigrams)",
                             "Two-word phrases (bigrams)"],
                    key=f"{KP}prof_tk_ngrams",
                    help="Phrases preserve concepts that single words split apart "
                         "(e.g., 'data science' as a unit, not 'data' + 'science'). "
                         "N-grams don't cross subtitle punctuation, and stopwords "
                         "are removed before phrases are built. Selecting more sizes "
                         "doesn't slow analysis meaningfully."
                )
                _ngram_map = {
                    "Single words (unigrams)": 1,
                    "Two-word phrases (bigrams)": 2,
                    "Three-word phrases (trigrams)": 3,
                }
                tk_ngram_sizes = tuple(sorted(
                    _ngram_map[c] for c in tk_ngram_choice
                )) or (1,)
                tk_min_freq = st.number_input(
                    "Min distinct titles (for bigrams/trigrams)",
                    min_value=1, max_value=20, value=2, step=1,
                    key=f"{KP}prof_tk_minfreq",
                    help="Multi-word phrases that appear in only one title are "
                         "usually noise (or a duplicate record). Set to 1 to "
                         "include them anyway. Doesn't apply to single-word "
                         "keywords."
                )
                tk_show_wordcloud = st.checkbox(
                    "Also show keyword word cloud", value=False, key=f"{KP}prof_tk_wc",
                    help="A separate cloud from the subject one — built from title "
                         "tokens, not subject headings. Uses unigrams only "
                         "(word clouds don't render multi-word phrases well)."
                )
                if tk_show_wordcloud:
                    tk_wc_max_words = st.slider(
                        "Cloud: max words", 20, 200, 100, 10, key=f"{KP}prof_tk_wc_max"
                    )
                    tk_wc_color = st.selectbox(
                        "Cloud: color scheme",
                        ["plasma", "viridis", "inferno", "magma", "cividis",
                         "twilight", "rainbow"],
                        key=f"{KP}prof_tk_wc_color"
                    )
                else:
                    tk_wc_max_words, tk_wc_color = 100, "plasma"
                tk_extra_raw = st.text_area(
                    "Extra stopwords (comma- or newline-separated)",
                    value="",
                    height=80,
                    key=f"{KP}prof_tk_stops",
                    help="Add domain-specific words to ignore — e.g., your "
                         "publisher's series name, or a recurring genre word that "
                         "isn't analytically interesting. Applied before n-grams "
                         "are built, so 'history of medicine' becomes the bigram "
                         "'history medicine' if 'of' is in the stopword list "
                         "(it already is)."
                )
                tk_extra_stopwords = {
                    w.strip().lower() for w in re.split(r"[,\n]", tk_extra_raw) if w.strip()
                }
                if tk_extra_stopwords:
                    st.caption(f"Filtering out {len(tk_extra_stopwords)} extra word(s).")
        else:
            tk_top_n, tk_show_wordcloud = 30, False
            tk_wc_max_words, tk_wc_color = 100, "plasma"
            tk_extra_stopwords = set()
            tk_ngram_sizes = (1, 2, 3)
            tk_min_freq = 2

        st.caption(
            "Changes here apply the next time you click **Re-run analysis** "
            "(or the main button below)."
        )
        rerun_clicked = st.button("🔄 Re-run analysis", key=f"{KP}prof_rerun", use_container_width=True)

    # Decide whether to run analysis:
    # - Always run on first upload of a file (no cached results for this file)
    # - Re-run if user clicks the rerun button inside the expander
    # - Re-run if user explicitly clicks the main Analyze button (shown below)
    file_key = _make_file_key(uploaded_file)
    last_run_key = st.session_state.get(f'{KP}prof_last_run_file_key')
    needs_autorun = st.session_state.get(f'{KP}prof_results') is None or last_run_key != file_key

    main_run_clicked = st.button(
        "🔍 Re-analyze collection",
        type="secondary",
        use_container_width=True,
        key=f"{KP}prof_run",
        help="Click to re-run with current settings."
    )

    if needs_autorun or rerun_clicked or main_run_clicked:
        w_key = '_weight' if use_weight else None
        # Structure mode is usage-free: never expose usage columns or
        # usage-weighted views even if the file happens to carry a usage column.
        eff_weight_col = weight_col if mode == "usage" else None
        pbar = st.progress(0, "Starting analysis...")
        results = _profiler_run_analysis(
            df, subj_col, lc_col, title_col, w_key, sel_classes, pbar,
            has_usage_col=bool(eff_weight_col),
            ngram_sizes=tk_ngram_sizes,
            lc_filter_active=lc_filter_active,
        )
        st.session_state[f'{KP}prof_results'] = results
        st.session_state[f'{KP}prof_last_run_file_key'] = file_key
        if sel_classes is not None and lc_col:
            mask = df['_lc_main'].isin(sel_classes)
            if not lc_filter_active:
                mask = mask | df['_lc_main'].isna()
            st.session_state[f'{KP}prof_filtered_idx'] = df.index[mask]
        else:
            st.session_state[f'{KP}prof_filtered_idx'] = df.index
        st.session_state[f'{KP}prof_settings'] = {
            'weight_label': weight_label if use_weight else 'Title Count',
            'usage_col_label': eff_weight_col or 'Usage',
            'has_usage_col': bool(eff_weight_col),
            'top_n_subjects': top_n,
            'show_sunburst': show_sunburst, 'show_treemap': show_treemap,
            'show_subject_bars': show_bars,
            'show_wordcloud': show_wordcloud,
            'wc_max_words': wc_max_words, 'wc_min_len': wc_min_len, 'wc_color': wc_color,
            'show_heatmap': show_heatmap,
            'show_title_keywords': show_title_keywords,
            'tk_top_n': tk_top_n, 'tk_show_wordcloud': tk_show_wordcloud,
            'tk_wc_max_words': tk_wc_max_words, 'tk_wc_color': tk_wc_color,
            'tk_extra_stopwords': tk_extra_stopwords,
            'tk_ngram_sizes': tk_ngram_sizes, 'tk_min_freq': tk_min_freq,
            'show_coverage_vs_use': show_coverage_vs_use,
            'cvu_over': cvu_over, 'cvu_under': cvu_under,
            'cvu_min_titles': cvu_min_titles, 'cvu_show_sub': cvu_show_sub,
            'show_gap_analysis': show_gap, 'show_detail_table': show_detail,
        }
        pbar.empty()

    if st.session_state[f'{KP}prof_results']:
        _profiler_display_results(
            st.session_state[f'{KP}prof_results'],
            st.session_state.get(f'{KP}prof_settings', {
                'weight_label': 'Title Count', 'usage_col_label': 'Usage',
                'top_n_subjects': 30,
                'show_sunburst': True, 'show_treemap': True,
                'show_subject_bars': True, 'show_wordcloud': True,
                'wc_max_words': 100, 'wc_min_len': 3, 'wc_color': 'viridis',
                'show_heatmap': True,
                'show_title_keywords': False,
                'tk_top_n': 30, 'tk_show_wordcloud': False,
                'tk_wc_max_words': 100, 'tk_wc_color': 'plasma',
                'tk_extra_stopwords': set(),
                'tk_ngram_sizes': (1, 2, 3), 'tk_min_freq': 2,
                'show_coverage_vs_use': False,
                'cvu_over': 2.0, 'cvu_under': 0.5,
                'cvu_min_titles': 10, 'cvu_show_sub': True,
                'show_gap_analysis': True, 'show_detail_table': False,
            }),
            df,
            st.session_state.get(f'{KP}prof_filtered_idx', df.index),
            title_col=title_col,
            weight_col=(weight_col if mode == "usage" else None),
            author_col=author_col,
            date_col=date_col,
            location_col=location_col,
            subj_col=subj_col,
            lc_col=lc_col,
        )


def page_collection_profiler():
    """Collection Profiler page — holdings structure only (no usage views)."""
    _profiler_ui(mode="structure")


def page_use_analysis():
    """Use Analysis — one tool for all usage-driven analysis, print and electronic.

    Branches by data type:
      • Print circulation (subject + usage) -> profiler engine, usage on
      • Electronic / COUNTER 5 report       -> the COUNTER reader (native file)
      • Electronic / other usage data       -> profiler engine, usage on (generic)

    The print and other-usage branches expect the synced explicit-zero master
    from the Zero-Use Identifier so titles with no recorded use count as 0.
    """
    st.header("\U0001F4C8 Use Analysis")
    st.markdown(
        "**What's getting used \u2014 and is it worth keeping?** One place for "
        "usage-driven analysis across print and electronic. Most branches expect "
        "the synced **explicit-zero master** from the Zero-Use Identifier, so "
        "titles with no recorded use are counted as 0 rather than dropped."
    )
    data_type = st.radio(
        "What kind of usage data are you analyzing?",
        ["Print circulation (subject + usage)",
         "Electronic \u2014 COUNTER 5 report",
         "Electronic \u2014 other usage data"],
        index=0, key="use_data_type",
    )
    st.markdown("---")

    if data_type.startswith("Electronic \u2014 COUNTER"):
        st.info(
            "\u26A0\uFE0F COUNTER 5 reports list only titles that were used. Cost-per-use "
            "and dead-weight percentages need the full title universe as the "
            "denominator. Before relying on those figures, reconcile the vendor's "
            "Portfolio List (Alma) or supplied title list against this report in the "
            "**Zero-Use Identifier** first \u2014 the native COUNTER file below still "
            "drives the monthly-trend and per-title views."
        )
        _render_counter_mode()
    elif data_type.startswith("Print"):
        _profiler_ui(mode="usage", flavor="print")
    else:
        _profiler_ui(mode="usage", flavor="other")


# =====================================================================
# =====================================================================
# TOOL 2: COUNTER ANALYZER
# =====================================================================
# "Which e-resources are pulling their weight?"
# Formal COUNTER 5 reports only — TR/TR_J3/TR_B1/DR/PR/IR with the
# 12–13 row metadata header and monthly columns. Vendor admin exports
# (e.g., EBSCO Detailed Report with subjects/LC) belong in the Profiler.
# Print circulation now lives in the Profiler's Title Analysis tab.
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
    """Load a print circulation CSV. Tolerates a leading '#' provenance block
    (e.g. when re-feeding the Zero-Use explicit-zero master)."""
    skip = _count_leading_comment_lines(file_bytes)
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding='utf-8-sig', low_memory=False,
                         skiprows=skip)
    except Exception:
        df = pd.read_csv(BytesIO(file_bytes), encoding='latin-1', low_memory=False,
                         skiprows=skip)
    df.columns = df.columns.str.strip()
    return df


def _identify_month_columns(df):
    """Identify columns that look like 'Jan-2025' or 'Jan 2025'."""
    return [c for c in df.columns
            if re.match(r'^[A-Za-z]{3}[- ]\d{4}$', c)]


def _month_col_to_date(col):
    """Convert 'Jan-2025' or 'Jan 2025' to a pd.Timestamp (first of month)."""
    try:
        return pd.to_datetime(col.replace(' ', '-'), format='%b-%Y')
    except Exception:
        return pd.NaT


def _detect_counter_date_range(month_cols):
    """From a list of month columns, return (min_date, max_date, sorted_cols)
    or (None, None, []) if empty.
    """
    if not month_cols:
        return None, None, []
    dated = [(c, _month_col_to_date(c)) for c in month_cols]
    dated = [(c, d) for c, d in dated if pd.notna(d)]
    if not dated:
        return None, None, []
    dated.sort(key=lambda x: x[1])
    return dated[0][1], dated[-1][1], [c for c, _ in dated]


DATE_COL_ALIASES = [
    'Checkout Date', 'checkout_date', 'Loan Date', 'loan_date',
    'Transaction Date', 'transaction_date', 'Last Charge Date', 'last_charge_date',
    'Last Used', 'last_used', 'Last Checkout', 'last_checkout',
    'Date', 'date', 'DATE', 'Due Date', 'due_date',
    'File Last View Date', 'Last View Date', 'last_view_date',
]


def _detect_print_date_column(df):
    """Find a date-like column in a print circulation dataframe.

    Returns the column name or None. Tries aliases first, then falls back
    to any column whose name contains date-related keywords AND parses as dates
    for >50% of values. Numeric year-only columns (e.g., "Loan Year" = 2025)
    are accepted as date columns when the values are plausibly years.
    """
    # Try explicit aliases (exact match, case-insensitive)
    for alias in DATE_COL_ALIASES:
        for col in df.columns:
            if alias.lower() == col.lower():
                return col

    # Year-only columns: numeric, named with 'year', values in plausible year range
    for col in df.columns:
        lc = col.lower()
        if 'year' in lc and pd.api.types.is_numeric_dtype(df[col]):
            try:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    # Plausible years: between 1500 and the current year + 2
                    in_range = ((vals >= 1500) & (vals <= 2100)).sum()
                    if in_range / max(1, len(vals)) > 0.9:
                        return col
            except Exception:
                continue

    # Partial match — must contain date-related keyword AND parse as dates
    date_keywords = ['date', 'checkout', 'loan', 'charge']
    for col in df.columns:
        lc = col.lower()
        if any(tok in lc for tok in date_keywords):
            # Skip numeric columns — pd.to_datetime coerces numbers to epoch dates
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() / max(1, len(df)) > 0.5:
                    return col
            except Exception:
                continue
    return None


def _format_date_range(start, end):
    """Format a date range for display: 'Jan 2025 – Jun 2025' or 'Feb 12 – Mar 30, 2025'."""
    if pd.isna(start) or pd.isna(end):
        return "unknown"
    if start.year == end.year and start.month == end.month:
        return start.strftime('%B %Y')
    if start.year == end.year:
        return f"{start.strftime('%b')} – {end.strftime('%b %Y')}"
    return f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}"


def _slug_period(label):
    """Turn a period label like 'Jan – Jun 2025' into a filename-safe 'Jan-Jun-2025'."""
    if not label or label == "unknown period":
        return ""
    slug = re.sub(r'[^\w\s-]', '', label)  # drop punctuation
    slug = re.sub(r'\s+', '-', slug.strip())  # spaces → dashes
    return slug


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
            The loader auto-skips the metadata header and detects the reporting period
            from month columns.

            **Typical workflow:**
            1. Load the file and pick a metric (usually `Unique_Item_Requests`)
            2. Check the detected reporting period at the top; narrow it with the
               **Date Range** filter in the sidebar if you want to focus on a specific window
            3. Review the **Top Titles** tab to see your workhorses
            4. Check **Cancellation Review** for underperformers
            5. Use **Publisher Summary** to evaluate whole packages
            6. Look at **Monthly Trends** to spot seasonality or decline

            **Tip:** Set a custom reporting-period label in the sidebar (e.g., "FY2025 Q1")
            to label downloads clearly.
            """)
        return

    try:
        # Check session cache first
        cached = _cached_df_for_tool("usage_counter", uploaded_file)
        if cached is not None:
            df_raw = cached['df']
            skip_used = cached['skip_used']
            st.success(f"✅ Using cached data for *{uploaded_file.name}* "
                       f"({len(df_raw):,} rows)")
        else:
            file_bytes = uploaded_file.getvalue()
            df_raw, skip_used = _load_counter_csv(file_bytes, uploaded_file.name)
            _store_cached_df("usage_counter", uploaded_file,
                             {'df': df_raw, 'skip_used': skip_used})
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

        # Date range detection from month columns
        date_min, date_max, sorted_month_cols = _detect_counter_date_range(month_cols)

        # Reporting period banner
        if date_min is not None:
            period_label = _format_date_range(date_min, date_max)
            st.info(f"📅 **Reporting period detected:** {period_label} "
                    f"({len(sorted_month_cols)} months of data)")
        else:
            st.warning("📅 No month columns detected. The file may have a non-standard "
                       "format, or may contain only aggregate totals.")
            period_label = "unknown period"

        with st.expander("🔍 Column Detection", expanded=False):
            st.write(f"**Total column:** `{total_col}`")
            st.write(f"**Month columns detected:** {len(month_cols)} "
                     f"({sorted_month_cols[0] if sorted_month_cols else 'none'} … "
                     f"{sorted_month_cols[-1] if sorted_month_cols else 'none'})")
            st.write(f"**All columns:** {list(df_raw.columns)}")

        if total_col is None:
            st.error("❌ Could not find a 'Reporting Period Total' column. "
                     "This may not be a standard COUNTER 5 TR file.")
            return

        # Sidebar filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔎 Counter Filters")

        # Date range filter (uses month columns)
        use_date_filter = False
        selected_month_cols = sorted_month_cols  # default: all months
        if sorted_month_cols and len(sorted_month_cols) > 1:
            with st.sidebar.expander("📅 Date Range", expanded=False):
                use_date_filter = st.checkbox(
                    "Filter by date range", value=False, key="usage_use_dates"
                )
                if use_date_filter:
                    # Use select_slider over sorted month labels (simpler than date picker
                    # for monthly granularity, and avoids confusion with non-month days)
                    start_label, end_label = st.select_slider(
                        "Months to include",
                        options=sorted_month_cols,
                        value=(sorted_month_cols[0], sorted_month_cols[-1]),
                        key="usage_date_range"
                    )
                    start_idx = sorted_month_cols.index(start_label)
                    end_idx = sorted_month_cols.index(end_label)
                    selected_month_cols = sorted_month_cols[start_idx:end_idx + 1]
                    sel_start = _month_col_to_date(start_label)
                    sel_end = _month_col_to_date(end_label)
                    period_label = _format_date_range(sel_start, sel_end)
                    st.caption(f"Selected: **{period_label}** "
                               f"({len(selected_month_cols)} months)")

        # Override label for reports/downloads
        with st.sidebar.expander("🏷️ Report label (optional)", expanded=False):
            custom_period = st.text_input(
                "Custom reporting period label",
                value="",
                placeholder=f"e.g., FY2025 Q1–Q2",
                help="Overrides the auto-detected label on downloads and headers.",
                key="usage_period_label"
            )
            if custom_period.strip():
                period_label = custom_period.strip()

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

        # KPIs — if date-filtered, recompute totals from selected month columns
        st.markdown("---")
        st.subheader(f"Overview — {selected_metric} · {period_label}")

        # Compute the per-row total that respects the date filter
        if use_date_filter and selected_month_cols:
            # Sum just the selected months (coerce to numeric first)
            month_data = df_filtered[selected_month_cols].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(0)
            df_filtered = df_filtered.copy()
            df_filtered['_total'] = month_data.sum(axis=1)
        else:
            df_filtered = df_filtered.copy()
            df_filtered['_total'] = pd.to_numeric(
                df_filtered[total_col], errors='coerce'
            ).fillna(0)

        total_usage = df_filtered['_total'].sum()
        unique_titles = df_filtered['Title'].nunique() if 'Title' in df_filtered.columns else 0
        avg_usage = total_usage / unique_titles if unique_titles > 0 else 0
        zero_use = (df_filtered['_total'] == 0).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Usage", f"{int(total_usage):,}")
        k2.metric("Unique Titles", f"{unique_titles:,}")
        k3.metric("Avg Usage / Title", f"{avg_usage:.1f}")
        k4.metric("Zero-Use Titles", f"{zero_use:,}")

        # Notes — shown above tabs so users annotate *before* downloading
        notes = _notes_widget(
            "usage_counter",
            placeholder="e.g., Prepared for FY2025 e-resource renewal review. "
                        "Cancellation candidates flagged for follow-up with Anthony."
        )

        # Fresh tray for this render pass
        _reset_tray("usage_counter")

        # Analysis tabs
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "Top titles", "Cancellation review",
            "Publisher summary", "Monthly trends"
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
            _top_fname = f"top_titles_{_slug_period(period_label)}.csv".replace('_.', '.')
            _top_bytes = _annotate_csv(top_titles, notes,
                                       extra_meta={'Tool': 'COUNTER Analyzer',
                                                   'View': 'Top Titles',
                                                   'Metric': selected_metric,
                                                   'Period': period_label})
            st.download_button("📥 Top titles (CSV)", _top_bytes, _top_fname,
                               "text/csv", key="usage_dl_top")
            _add_to_tray("usage_counter", _top_fname, _top_bytes)

        with tab2:
            st.info("Review titles with low usage for potential cancellation or renegotiation.")
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
            _cancel_fname = f"cancellation_review_{_slug_period(period_label)}.csv".replace('_.', '.')
            _cancel_bytes = _annotate_csv(low_use_df, notes,
                                          extra_meta={'Tool': 'COUNTER Analyzer',
                                                      'View': 'Cancellation Review',
                                                      'Metric': selected_metric,
                                                      'Threshold': threshold,
                                                      'Period': period_label})
            st.download_button("📥 Cancellation review list (CSV)",
                               _cancel_bytes, _cancel_fname, "text/csv",
                               key="usage_dl_cancel")
            _add_to_tray("usage_counter", _cancel_fname, _cancel_bytes)

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
                _pub_fname = f"publisher_summary_{_slug_period(period_label)}.csv".replace('_.', '.')
                _pub_bytes = _annotate_csv(pub_summary, notes,
                                           extra_meta={'Tool': 'COUNTER Analyzer',
                                                       'View': 'Publisher Summary',
                                                       'Metric': selected_metric,
                                                       'Period': period_label})
                st.download_button("📥 Publisher summary (CSV)",
                                   _pub_bytes, _pub_fname, "text/csv",
                                   key="usage_dl_pub")
                _add_to_tray("usage_counter", _pub_fname, _pub_bytes)
            else:
                st.info("No Publisher column in this file.")

        with tab4:
            if month_cols:
                # Use selected_month_cols (which equals sorted_month_cols when filter off)
                id_vars = ['Title']
                if 'Publisher' in df_filtered.columns:
                    id_vars.append('Publisher')
                present_months = [c for c in selected_month_cols if c in df_filtered.columns]
                if not present_months:
                    st.info("No month columns found in filtered data.")
                else:
                    df_melted = df_filtered.melt(
                        id_vars=id_vars, value_vars=present_months,
                        var_name='Month', value_name='Usage'
                    )
                    df_melted['Usage'] = pd.to_numeric(df_melted['Usage'], errors='coerce').fillna(0)
                    df_melted['Month'] = pd.to_datetime(
                        df_melted['Month'].str.replace(' ', '-'), format='%b-%Y', errors='coerce'
                    )
                    monthly_trend = df_melted.groupby('Month', as_index=False)['Usage'].sum()
                    fig_trend = px.line(
                        monthly_trend, x='Month', y='Usage', markers=True,
                        title=f"Monthly {selected_metric} — {period_label}",
                    )
                    fig_trend.update_traces(line_color='#285C4D')
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # Top-title breakdown
                    st.markdown("**Monthly trend for top 5 titles:**")
                    top5_titles = df_filtered.nlargest(5, '_total')['Title'].tolist()
                    tdf = df_melted[df_melted['Title'].isin(top5_titles)]
                    if not tdf.empty:
                        fig_tt = px.line(
                            tdf.groupby(['Month', 'Title'], as_index=False)['Usage'].sum(),
                            x='Month', y='Usage', color='Title', markers=True,
                            title=f"Monthly Usage — Top 5 Titles — {period_label}"
                        )
                        st.plotly_chart(fig_tt, use_container_width=True)
            else:
                st.info("No month columns (e.g., `Jan-2025`) detected in this file.")

        # Consolidated download tray
        st.markdown("---")
        st.subheader("Downloads")
        _render_download_tray("usage_counter",
                              zip_filename=f"usage_counter_{_slug_period(period_label)}.zip".replace('_.', '.'))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("Ensure you are uploading a standard COUNTER 5 TR CSV. "
                "If the file has an unusual structure, try exporting a fresh copy from the vendor.")


def page_counter_analyzer():
    """Tool 2: COUNTER Analyzer.

    Handles formal COUNTER 5 reports (TR/TR_J3/TR_B1/DR/PR/IR) with the
    standard 12–13 row metadata header and monthly columns. The standard
    COUNTER 5 spec doesn't carry subject data, so this tool focuses on
    title-level usage triage: top titles, cancellation review, publisher
    rollups, and monthly trends.

    Files that are *vendor admin exports* rather than formal COUNTER reports
    (e.g., EBSCO's "Detailed Report" with subjects and LC) belong in the
    Collection Profiler instead — they have catalog-style metadata that the
    Profiler can analyze across LC, Subject, and Title views.
    """
    st.header("📊 COUNTER Analyzer")
    st.markdown(
        "**Which e-resources are pulling their weight?** "
        "Title-level usage analysis from formal COUNTER 5 reports — for "
        "renewal decisions, cancellation candidates, Big Deal evaluation, "
        "and package-vs-title comparisons."
    )
    with st.expander("ℹ️ When to use this tool"):
        st.markdown(
            "- **Renewal review (Sept):** Run on each vendor's TR before submitting "
            "renewal changes. Focus on Publisher Summary and Cancellation Review tabs.\n"
            "- **Cancellation prep (July):** Pull most recent 12 months of TR_J3 to "
            "identify low-use titles for the Aug 1 deadline.\n"
            "- **Big Deal evaluation (Dec):** Run on each Big Deal under renewal "
            "to surface dead-weight titles as bargaining leverage.\n"
            "- **Annual watchlist (May):** Run on full prior-calendar-year COUNTER "
            "data as part of the year's biggest evidence harvest.\n\n"
            "**Not for:** vendor admin exports (e.g., EBSCO Detailed Report) — "
            "those have subjects and LC, and belong in the Collection Profiler."
        )

    _render_counter_mode()


# Backward-compatibility alias: prior sessions or bookmarks pointed at
# "Usage & Subscription Analyzer". Forward to the renamed COUNTER analyzer.
def page_usage_analyzer():
    """Deprecated wrapper — Usage Analyzer was renamed to COUNTER Analyzer."""
    page_counter_analyzer()



# =====================================================================
# =====================================================================
# TOOL 3: ZERO-USE IDENTIFIER
# =====================================================================
# "What do we own that isn't being used at all?"
# Two-file comparison: holdings universe vs. usage report.
# Surfaces titles in holdings that don't appear in usage — the inverse
# of the Usage Analyzer, which starts with what HAS been used.
# =====================================================================
# =====================================================================

# ---- Identifier normalization helpers ----

def _normalize_isbn(val):
    """Strip hyphens, spaces, and non-digits (except trailing X for ISBN-10).
    Returns the canonical form, or None if not a plausible ISBN."""
    if pd.isna(val):
        return None
    s = str(val).upper().strip()
    s = re.sub(r'[^\dX]', '', s)
    if len(s) not in (10, 13):
        # Some files cram multiple ISBNs in one cell; try first 13 then 10
        for length in (13, 10):
            if len(s) >= length:
                candidate = s[:length]
                if length == 10 or candidate.startswith(('978', '979')):
                    return candidate
        return None
    return s


def _normalize_issn(val):
    """ISSNs are 8 chars; canonical form has hyphen but we strip it for matching."""
    if pd.isna(val):
        return None
    s = str(val).upper().strip()
    s = re.sub(r'[^\dX]', '', s)
    return s if len(s) == 8 else None


def _normalize_doi(val):
    """Lowercase, strip leading 'doi:' or URL prefix, trim whitespace."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    s = re.sub(r'^https?://(dx\.)?doi\.org/', '', s)
    s = re.sub(r'^doi:\s*', '', s)
    return s if s and '/' in s else None  # Real DOIs always have a slash


def _normalize_oclc(val):
    """OCLC numbers — strip prefixes like 'ocm', 'ocn', '(OCoLC)', leave digits."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r'^\(OCoLC\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(ocm|ocn|on)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\D', '', s)
    return s if s else None


def _build_title_author_key(title, author):
    """Composite key for fallback matching when identifiers are absent.
    Uses normalized title + first significant word of author."""
    t = normalize_text(title) if title else ""
    if not t:
        return None
    a = normalize_text(author) if author else ""
    # First token of normalized author — for "Smith, John" → "smith"
    a_tok = a.split()[0] if a else ""
    return f"{t}|{a_tok}" if t else None


def _detect_id_columns(df):
    """Detect identifier columns in a dataframe."""
    return {
        'isbn': find_column(df, ISBN_ALIASES),
        'issn': find_column(df, ISSN_ALIASES),
        'doi': find_column(df, DOI_ALIASES),
        'oclc': find_column(df, OCLC_ALIASES),
        'title': find_column(df, TITLE_ALIASES),
        'author': find_column(df, AUTHOR_ALIASES),
    }


def _build_match_keys(df, id_cols):
    """Add normalized matching key columns to a dataframe in place.
    Returns the list of (key_type, column_name) actually built."""
    built = []
    if id_cols.get('isbn'):
        df['_key_isbn'] = df[id_cols['isbn']].apply(_normalize_isbn)
        built.append(('isbn', '_key_isbn'))
    if id_cols.get('issn'):
        df['_key_issn'] = df[id_cols['issn']].apply(_normalize_issn)
        built.append(('issn', '_key_issn'))
    if id_cols.get('doi'):
        df['_key_doi'] = df[id_cols['doi']].apply(_normalize_doi)
        built.append(('doi', '_key_doi'))
    if id_cols.get('oclc'):
        df['_key_oclc'] = df[id_cols['oclc']].apply(_normalize_oclc)
        built.append(('oclc', '_key_oclc'))
    if id_cols.get('title'):
        author_series = df[id_cols['author']] if id_cols.get('author') else pd.Series([None] * len(df), index=df.index)
        df['_key_titleauth'] = [_build_title_author_key(t, a)
                                 for t, a in zip(df[id_cols['title']], author_series)]
        built.append(('title+author', '_key_titleauth'))
    return built


def _match_holdings_to_usage(holdings_df, usage_df, holdings_keys, usage_keys,
                              usage_weight_col=None, usage_carry_cols=None):
    """Cascade-match each holdings row against the usage file.

    Returns the holdings_df with two new columns:
        _matched_via : str (which key matched, or 'unmatched')
        _usage_total : float (sum of usage from matched usage rows, or 0)

    If `usage_carry_cols` is given, each of those usage-file columns (e.g.
    Subjects, LC when they live on the usage side) is also carried onto the
    matched holdings rows. Carried values are taken from the matched usage
    row(s); unmatched / zero-use rows are left blank, because by definition
    they have no usage row to read metadata from.

    Matching cascades by reliability: ISBN > DOI > OCLC > ISSN > title+author.
    A holdings row counts as 'matched' on the first key that finds at least
    one usage row.

    Returns (matched_df, carry_out_names) where carry_out_names maps each
    requested usage column to the output column name actually used (renamed to
    avoid clobbering an identically named holdings column).
    """
    holdings_df = holdings_df.copy()
    holdings_df['_matched_via'] = 'unmatched'
    holdings_df['_usage_total'] = 0.0

    carry_cols = [c for c in (usage_carry_cols or []) if c in usage_df.columns]
    carry_out = {}
    for c in carry_cols:
        out = c if c not in holdings_df.columns else f"{c} (from usage)"
        carry_out[c] = out
        holdings_df[out] = pd.NA

    # Build per-key lookup tables from usage side: key_value → total_usage,
    # plus (optionally) key_value → {carry_col: first non-null value}.
    usage_lookups = {}
    carry_lookups = {}
    for key_type, key_col in usage_keys:
        if key_col not in usage_df.columns:
            continue
        valid = usage_df[usage_df[key_col].notna() & (usage_df[key_col] != '')]
        if valid.empty:
            continue
        if usage_weight_col and usage_weight_col in valid.columns:
            grouped = valid.groupby(key_col)[usage_weight_col].sum()
        else:
            # No weight column → just count rows per key
            grouped = valid.groupby(key_col).size().astype(float)
        usage_lookups[key_type] = grouped.to_dict()
        if carry_cols:
            # First non-null value of each carry column per key
            carry_lookups[key_type] = (
                valid.groupby(key_col)[carry_cols].first().to_dict('index')
            )

    # Identifiers first (most reliable), title+author last
    priority = ['isbn', 'doi', 'oclc', 'issn', 'title+author']
    for key_type in priority:
        h_col = next((kc for kt, kc in holdings_keys if kt == key_type), None)
        if h_col is None or key_type not in usage_lookups:
            continue
        lookup = usage_lookups[key_type]
        clook = carry_lookups.get(key_type, {})
        # Only fill rows still unmatched — keeps higher-priority matches sticky
        unmatched_mask = holdings_df['_matched_via'] == 'unmatched'
        h_values = holdings_df.loc[unmatched_mask, h_col]
        for idx, val in h_values.items():
            if val and val in lookup:
                holdings_df.at[idx, '_matched_via'] = key_type
                holdings_df.at[idx, '_usage_total'] = lookup[val]
                if val in clook:
                    row = clook[val]
                    for c in carry_cols:
                        v = row.get(c)
                        if pd.notna(v):
                            holdings_df.at[idx, carry_out[c]] = v

    return holdings_df, carry_out


def page_zero_use_identifier():
    """Tool 4: Zero-Use Identifier — compare a holdings list to a usage list."""
    st.header("🔍 Zero-Use Identifier")
    st.markdown(
        "**What do we own that isn't being used?** Compare a list of all your "
        "holdings against a usage report to surface titles, journals, or "
        "databases that aren't appearing in usage at all."
    )
    with st.expander("ℹ️ When to use this tool"):
        st.markdown(
            "- **Collections:** Identify dead-weight items in any format (print, "
            "e-books, e-journals, databases, streaming media). Especially powerful "
            "for e-resources where 'zero use' is invisible in the usage report itself "
            "(many vendors omit titles with no use entirely).\n"
            "- **Cancellation prep:** Combine with the **COUNTER Analyzer** — "
            "use this tool to find titles missing from usage altogether, then use "
            "COUNTER to triage low-but-nonzero items.\n"
            "- **Off-site storage:** Surface print holdings with no circulation, "
            "filtered by pub-year cutoff so newer items don't get flagged unfairly.\n"
            "- **Renewal evidence:** Show admin or faculty exactly which package "
            "titles haven't been used at all."
        )

    with st.expander("📖 How matching works", expanded=False):
        st.markdown("""
        Two files, one job: **what's in holdings but missing from usage?**

        **Matching cascade** (most reliable first):
        1. **ISBN / ISSN / DOI / OCLC** — exact match after normalization
           (strips hyphens, prefixes like `(OCoLC)`, URL wrappers on DOIs, etc.)
        2. **Title + first author word** — fallback when no shared identifier
           exists. Uses the same text normalization as the rest of the dashboard
           (lowercase, accent-strip, punctuation collapse).

        **Holdings file** = the universe (full catalog, e-journal A-to-Z list,
        database list, etc.). At minimum needs a Title column; identifier
        columns improve match quality dramatically.

        **Usage file** = anything with a count attached. COUNTER reports,
        circulation exports, link-resolver clicks — the same kinds of files
        the COUNTER Analyzer and Profiler's Title Analysis tab accept.

        **Match quality:** the result table includes a `Matched Via` column so
        you can spot-check whether a fallback title-match looks right.

        **The unmatched ≠ zero-use caveat:** If your usage report only includes
        titles that had at least one use (very common — many vendors omit
        zero-use rows), then unmatched items are genuinely zero-use. If your
        usage report includes zero-use rows explicitly, then unmatched items
        might just be metadata mismatches. The two populations are split into
        separate tabs so you can decide.
        """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📚 Holdings file** — the universe")
        holdings_file = st.file_uploader(
            "Upload holdings CSV", type=['csv'], key="zu_holdings_upload",
            help="Your full title list, e-journal A-Z, database list, etc."
        )
    with c2:
        st.markdown("**📊 Usage file** — what's been used")
        usage_file = st.file_uploader(
            "Upload usage CSV", type=['csv'], key="zu_usage_upload",
            help="COUNTER report, circulation export, or any list with a count column."
        )

    if not (holdings_file and usage_file):
        st.info("Upload **both** files to begin. The holdings file is your "
                "full list; the usage file is what has been used.")
        return

    try:
        # Load both files (with caching)
        cached_h = _cached_df_for_tool("zu_holdings", holdings_file)
        cached_u = _cached_df_for_tool("zu_usage", usage_file)

        if cached_h is not None:
            holdings_df = cached_h.copy()
        else:
            holdings_df = _load_print_csv(holdings_file.getvalue(), holdings_file.name)
            _store_cached_df("zu_holdings", holdings_file, holdings_df)

        if cached_u is not None:
            usage_df = cached_u.copy()
        else:
            usage_df = _load_print_csv(usage_file.getvalue(), usage_file.name)
            _store_cached_df("zu_usage", usage_file, usage_df)

        st.success(
            f"✅ Loaded **{len(holdings_df):,}** holdings rows and "
            f"**{len(usage_df):,}** usage rows."
        )

        # Detect columns in both files
        h_ids = _detect_id_columns(holdings_df)
        u_ids = _detect_id_columns(usage_df)
        h_lc = find_column(holdings_df, LC_ALIASES)
        h_subj = find_column(holdings_df, SUBJECT_ALIASES)
        h_loc = find_column(holdings_df, ['Location', 'Location Name', 'location',
                                          'Library', 'Branch'])
        h_format = find_column(holdings_df, ['Format', 'Material Type', 'Resource Type',
                                             'Type', 'format', 'Bibliographic Format'])
        h_pubyear = find_column(holdings_df, ['Publication Year', 'Pub Year', 'Year',
                                              'pub_year', 'Publication Date',
                                              'Date of Publication'])
        u_weight = find_column(usage_df, WEIGHT_ALIASES)
        # Subjects (and sometimes LC) often live on the usage file rather than
        # holdings — e.g. an Alma Analytics usage export or EBSCO Detailed Report.
        u_subj = find_column(usage_df, SUBJECT_ALIASES)
        u_lc = find_column(usage_df, LC_ALIASES)

        # Capture the user's original holdings columns now, before key-building
        # and matching add internal helpers. Every one of these is carried into
        # the outputs so nothing from the source file — subjects, call numbers,
        # or any other metadata — is silently dropped by the curated projection.
        orig_holdings_cols = [c for c in holdings_df.columns
                              if not str(c).startswith('_')]

        with st.expander("🔍 Column detection & overrides", expanded=False):
            st.markdown("**Holdings file:**")
            hcols_text = " · ".join([f"{k.upper()}: `{v}`" if v else f"{k.upper()}: —"
                                     for k, v in h_ids.items()])
            st.caption(hcols_text)
            st.caption(f"LC: `{h_lc}` · Subjects: `{h_subj}` · Location: `{h_loc}` · "
                       f"Format: `{h_format}` · Pub Year: `{h_pubyear}`")

            st.markdown("**Usage file:**")
            ucols_text = " · ".join([f"{k.upper()}: `{v}`" if v else f"{k.upper()}: —"
                                     for k, v in u_ids.items()])
            st.caption(ucols_text)
            st.caption(f"Usage metric: `{u_weight}` · Subjects: `{u_subj}` · LC: `{u_lc}`")

            none_opt = "— count rows (no weighting) —"
            usage_cols = [none_opt] + list(usage_df.columns)
            default_idx = usage_cols.index(u_weight) if u_weight in usage_df.columns else 0
            new_weight = st.selectbox(
                "Override usage metric column",
                usage_cols, index=default_idx, key="zu_weight_override"
            )
            u_weight = None if new_weight == none_opt else new_weight

        # Validate: need at least Title in holdings, or one identifier
        if not h_ids.get('title') and not any(h_ids.get(k) for k in ('isbn', 'issn', 'doi', 'oclc')):
            st.error("❌ Holdings file needs at least a **Title** column or one "
                     "identifier column (ISBN, ISSN, DOI, OCLC). Couldn't find any.")
            return

        # Find shared key types between the two files
        shared_keys = []
        for k in ('isbn', 'issn', 'doi', 'oclc'):
            if h_ids.get(k) and u_ids.get(k):
                shared_keys.append(k.upper())
        if h_ids.get('title') and u_ids.get('title'):
            shared_keys.append('Title+Author')

        if not shared_keys:
            st.error("❌ No matchable columns found in common between the two files. "
                     "Need at least one of ISBN, ISSN, DOI, OCLC, or Title in both.")
            return

        st.info(f"🔗 Will match using: **{', '.join(shared_keys)}** "
                f"(cascading from most reliable to fallback)")

        # Coerce usage weight to numeric (or use row counts if no weight col)
        if u_weight and u_weight in usage_df.columns:
            usage_df['_weight'] = pd.to_numeric(usage_df[u_weight], errors='coerce').fillna(0)
            weight_for_match = '_weight'
        else:
            weight_for_match = None

        # Build match keys on both sides
        h_keys = _build_match_keys(holdings_df, h_ids)
        u_keys = _build_match_keys(usage_df, u_ids)

        # Carry subjects/LC from the usage file onto matched rows.
        usage_carry_cols = [c for c in (u_subj, u_lc) if c]

        # Run match
        with st.spinner("Matching holdings against usage..."):
            matched, carry_out = _match_holdings_to_usage(
                holdings_df, usage_df, h_keys, u_keys,
                usage_weight_col=weight_for_match,
                usage_carry_cols=usage_carry_cols,
            )

        # Coalesce descriptive metadata from BOTH sides into one column each.
        # Holdings wins (so a zero-use title keeps its holdings subject/LC); blanks
        # are then filled from the usage-carried value (so a used title with no
        # holdings subject still gets one). The redundant carried column is dropped.
        def _coalesce_meta(primary, secondary):
            if (primary and secondary and primary != secondary
                    and primary in matched.columns and secondary in matched.columns):
                s = matched[primary].astype('object')
                blank = s.isna() | s.astype(str).str.strip().isin(['', 'nan', 'None', '<NA>'])
                matched.loc[blank, primary] = matched.loc[blank, secondary]
                matched.drop(columns=[secondary], inplace=True)
                return primary
            return primary or secondary

        subj_carry = carry_out.get(u_subj) if u_subj else None
        lc_carry = carry_out.get(u_lc) if u_lc else None
        subject_col_final = _coalesce_meta(h_subj, subj_carry)
        lc_col_final = _coalesce_meta(h_lc, lc_carry)
        # Effective LC column for filtering/extraction: the unified one
        eff_lc = lc_col_final

        # Sidebar filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔎 Zero-Use Filters")

        # LC filter
        if eff_lc:
            matched['_lc_main'] = matched[eff_lc].apply(extract_lc_prefix)
            lc_avail = sorted(matched['_lc_main'].dropna().unique())
            if lc_avail:
                lc_labels = [f"{c} – {LC_CLASSES.get(c, '?')}" for c in lc_avail]
                sel_lc_labels = st.sidebar.multiselect(
                    "LC Class", lc_labels, default=lc_labels, key="zu_lc_filter"
                )
                sel_lc = [l.split(' –')[0] for l in sel_lc_labels]
                matched = matched[matched['_lc_main'].isin(sel_lc) | matched['_lc_main'].isna()].copy()

        # Location filter
        if h_loc:
            locs = sorted(matched[h_loc].dropna().unique())
            if locs:
                sel_locs = st.sidebar.multiselect(
                    "Location", locs, default=locs, key="zu_loc_filter"
                )
                matched = matched[matched[h_loc].isin(sel_locs) | matched[h_loc].isna()].copy()

        # Format filter
        if h_format:
            fmts = sorted(matched[h_format].dropna().unique())
            if fmts:
                sel_fmts = st.sidebar.multiselect(
                    "Format", fmts, default=fmts, key="zu_fmt_filter"
                )
                matched = matched[matched[h_format].isin(sel_fmts) | matched[h_format].isna()].copy()

        # Threshold for "low use"
        threshold = st.sidebar.number_input(
            "Use threshold (≤ this = flagged)",
            min_value=0.0, value=0.0, step=1.0, key="zu_threshold",
            help="0 = strictly zero use. Raise this to also catch low-use items."
        )

        # Treat unmatched items as zero-use? Off by default (safer assumption);
        # turn on when the usage report is known to include zero-use rows
        # explicitly (then unmatched truly = zero-use, not a coverage gap).
        treat_unmatched_as_zero = st.sidebar.checkbox(
            "Treat unmatched items as zero-use",
            value=False, key="zu_combine_unmatched",
            help="OFF (default): unmatched items appear in their own tab. Use when "
                 "your usage report omits zero-use titles (most COUNTER reports do).\n\n"
                 "ON: unmatched items are merged into the zero/low-use list. Use only "
                 "when you trust your usage report's coverage — i.e., it explicitly "
                 "includes rows for unused titles. Spot-check the match preview "
                 "below before flipping this on for a final cancellation list."
        )

        # Optional pub-year cutoff
        pubyear_cutoff = None
        if h_pubyear:
            with st.sidebar.expander("📅 Optional: limit by pub year", expanded=False):
                use_year = st.checkbox("Only flag items older than:", value=False, key="zu_use_year")
                if use_year:
                    pubyear_cutoff = st.number_input(
                        "Published before",
                        min_value=1800, max_value=2030, value=2015,
                        step=1, key="zu_year_cutoff",
                        help="Newer items often haven't had time to circulate. "
                             "Setting a cutoff focuses on items old enough that "
                             "zero use is meaningful."
                    )
                    matched['_pubyear'] = pd.to_numeric(matched[h_pubyear], errors='coerce')

        # KPIs
        st.markdown("---")
        st.subheader("Matching summary")

        total_holdings = len(matched)
        n_matched = (matched['_matched_via'] != 'unmatched').sum()
        n_unmatched = total_holdings - n_matched
        matched_only = matched[matched['_matched_via'] != 'unmatched']
        n_matched_low = (matched_only['_usage_total'] <= threshold).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Holdings", f"{total_holdings:,}")
        k2.metric("Matched to Usage", f"{n_matched:,}",
                  f"{n_matched/max(1,total_holdings)*100:.1f}%")
        if treat_unmatched_as_zero:
            # Combined view: matched-low + unmatched are both treated as zero/low-use
            n_combined = n_matched_low + n_unmatched
            k3.metric(f"≤ {threshold:g} Use (combined)", f"{n_combined:,}",
                      help="Matched-low + unmatched items, treated together "
                           "since you've indicated unmatched = zero-use.")
            k4.metric("Unmatched (now in main list)", f"{n_unmatched:,}",
                      f"{n_unmatched/max(1,total_holdings)*100:.1f}%",
                      help="These items are now folded into the zero/low-use list.")
        else:
            k3.metric(f"≤ {threshold:g} Use (matched)", f"{n_matched_low:,}",
                      help="Items that joined to the usage file but had use ≤ threshold.")
            k4.metric("Unmatched", f"{n_unmatched:,}",
                      f"{n_unmatched/max(1,total_holdings)*100:.1f}%",
                      help="Items in holdings with no row in the usage file. Likely "
                           "zero-use, but could also indicate a coverage gap in the "
                           "usage report.")

        # Match-method breakdown
        method_counts = matched['_matched_via'].value_counts()
        if len(method_counts) > 1:
            st.caption(
                "**Match methods used:** " +
                " · ".join([f"{m}: {c:,}" for m, c in method_counts.items()])
            )

        # ---- Match preview ----
        # Show a sample of each match type with both holdings & usage rows side-by-side
        # so users can sanity-check before trusting the join. Especially valuable for
        # the title+author fallback, which is the highest-risk match type.
        title_match_count = int(method_counts.get('title+author', 0))
        with st.expander(
            f"🔍 Match preview — spot-check the joins"
            + (f" ({title_match_count:,} via title+author fallback — review!)"
               if title_match_count > 0 else ""),
            expanded=False
        ):
            st.caption(
                "Random samples from each match type. The **title+author** "
                "tab is the most important to review — it's the fallback when no "
                "shared identifier exists, and it's the most likely place for "
                "false positives (e.g., two unrelated books with similar titles "
                "and the same first-token last name)."
            )
            # Build a sample-per-method preview, max 10 rows per method
            preview_methods = [m for m in method_counts.index if m != 'unmatched']
            if not preview_methods:
                st.info("No matches to preview yet. Run with files that share an "
                        "identifier or title column.")
            else:
                # Default to title+author tab if it exists (highest-risk to review)
                preview_tab_labels = []
                for m in preview_methods:
                    n = int(method_counts[m])
                    flag = " ⚠️" if m == 'title+author' else ""
                    preview_tab_labels.append(f"{m} ({n:,}){flag}")
                preview_tabs = st.tabs(preview_tab_labels)
                # Build usage-side title/author lookups so we can show what we matched against
                u_title_col = u_ids.get('title')
                u_author_col = u_ids.get('author')
                # Map each key column on the usage side to the original Title/Author columns
                u_keys_by_type = dict(u_keys)
                for tab_obj, method in zip(preview_tabs, preview_methods):
                    with tab_obj:
                        sample = matched[matched['_matched_via'] == method]
                        sample_size = min(10, len(sample))
                        if sample_size == 0:
                            st.info("No items in this method.")
                            continue
                        sample = sample.sample(n=sample_size,
                                               random_state=42).reset_index(drop=True)
                        # Find the matching usage row(s) for each sample row,
                        # using the same key column that produced the match.
                        h_key_col = next((kc for kt, kc in h_keys if kt == method), None)
                        u_key_col = u_keys_by_type.get(method)
                        rows = []
                        for _, hrow in sample.iterrows():
                            h_title = (hrow.get(h_ids['title'], '—')
                                       if h_ids.get('title') else '—')
                            h_author = (hrow.get(h_ids['author'], '—')
                                        if h_ids.get('author') else '—')
                            # Find first matching usage row
                            u_title = u_author = '—'
                            if h_key_col and u_key_col:
                                key_val = hrow.get(h_key_col)
                                if pd.notna(key_val) and key_val:
                                    matches_in_usage = usage_df[
                                        usage_df[u_key_col] == key_val
                                    ]
                                    if not matches_in_usage.empty:
                                        first = matches_in_usage.iloc[0]
                                        if u_title_col:
                                            u_title = first.get(u_title_col, '—')
                                        if u_author_col:
                                            u_author = first.get(u_author_col, '—')
                            rows.append({
                                'Holdings title': str(h_title)[:80],
                                'Holdings author': str(h_author)[:50],
                                'Usage title': str(u_title)[:80],
                                'Usage author': str(u_author)[:50],
                                'Total Use': hrow.get('_usage_total', 0),
                            })
                        preview_df = pd.DataFrame(rows)
                        st.dataframe(preview_df, use_container_width=True,
                                     hide_index=True, height=min(400, 50 + 35 * len(preview_df)))
                        if method == 'title+author':
                            st.caption(
                                "💡 If the holdings and usage titles look like genuinely "
                                "different books, your fallback joins are unreliable. "
                                "Consider adding identifier columns to one or both files, "
                                "or only trusting matches via ISBN/ISSN/DOI/OCLC."
                            )

        # Notes — annotate before downloading
        notes = _notes_widget(
            "zero_use",
            placeholder="e.g., FY2025 e-journal cancellation review. "
                        "Holdings = A-Z list export 11/1; usage = COUNTER TR_J3 (Jan-Oct)."
        )

        # Fresh tray for this render pass
        _reset_tray("zero_use")

        # Build the zero-use result set.
        # Default (split): only MATCHED items with low use go in the main list.
        # Unmatched items live in their own tab so users can review them
        # separately. When the combine toggle is on, unmatched items are
        # folded in (treating them as zero-use items the report didn't return).
        if treat_unmatched_as_zero:
            zero_use = matched[
                ((matched['_matched_via'] != 'unmatched')
                 & (matched['_usage_total'] <= threshold))
                | (matched['_matched_via'] == 'unmatched')
            ].copy()
        else:
            zero_use = matched[
                (matched['_matched_via'] != 'unmatched')
                & (matched['_usage_total'] <= threshold)
            ].copy()
        if pubyear_cutoff is not None and '_pubyear' in zero_use.columns:
            zero_use = zero_use[zero_use['_pubyear'] < pubyear_cutoff]

        # Build display columns once (carried into both outputs, incl. subjects)
        display_cols = []
        if h_ids.get('title'):
            display_cols.append(h_ids['title'])
        if h_ids.get('author'):
            display_cols.append(h_ids['author'])
        for k in ('isbn', 'issn', 'doi', 'oclc'):
            if h_ids.get(k):
                display_cols.append(h_ids[k])
        if h_lc:
            display_cols.append(h_lc)
        if h_subj:
            display_cols.append(h_subj)
        # Unified subject/LC (holdings preferred, usage-filled) — ensure present
        # even when the column came only from the usage side.
        for c in (subject_col_final, lc_col_final):
            if c and c not in display_cols:
                display_cols.append(c)
        if h_loc:
            display_cols.append(h_loc)
        if h_format:
            display_cols.append(h_format)
        if h_pubyear:
            display_cols.append(h_pubyear)
        # Pass through every other original holdings column so nothing the user
        # supplied (subjects, call numbers, notes, edition, etc.) is dropped —
        # the recognized columns above just get ordered first.
        for c in orig_holdings_cols:
            if c not in display_cols:
                display_cols.append(c)
        # De-dup in case of overlap
        seen = set()
        display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]

        # ---------------------------------------------------------------
        # Outputs — exactly two title-level deliverables:
        #   1. Zero-use titles            -> the flagged zero/low-use list.
        #   2. All titles (explicit zeros) -> every holdings title with a
        #      numeric use value; titles with no recorded use carry an
        #      explicit 0. This is the synced universe you feed to the
        #      Collection Profiler (Coverage-vs-Use) or COUNTER triage.
        # Both outputs retain subject metadata when the holdings file has it.
        # Structural breakdowns (LC / format / age) intentionally live in the
        # Collection Profiler, which is built to map the collection; that keeps
        # this tool to one job — reconcile holdings against usage.
        # ---------------------------------------------------------------
        st.markdown("---")

        # Downloads serve as zipped CSVs: a 1M-row export compresses to a small
        # fraction of its raw size, so transfers stay reliable on memory-capped
        # hosts. The master (all titles, explicit zeros) is the default and the
        # handoff into Use Analysis. The zero-use-only list is opt-in because it
        # is just a filter on the master's Use Status column, so building it is
        # extra memory + transfer for something derivable from the master.
        also_zero_list = st.checkbox(
            "Also generate the zero-use-only list (separate download)",
            value=False, key="zu_gen_zerolist",
            help="The master file already contains the zero-use rows (tagged in "
                 "Use Status). Turn this on only if you also want a ready-made "
                 "zero-use-only file.")

        tab_labels = ["All titles (explicit zeros)"]
        if also_zero_list:
            tab_labels.append("Zero-use titles")
        out_tabs = st.tabs(tab_labels)

        cutoff_msg = (' AND were published before ' + str(pubyear_cutoff)
                      if pubyear_cutoff else '')

        # --- Primary output: All titles with explicit zeros (the master) ---
        # Every holdings title with a numeric use value (0 where unused). No
        # pub-year cutoff here: this is the full synced universe meant to feed
        # Coverage-vs-Use / COUNTER triage. The cutoff only narrows the optional
        # zero-use list.
        with out_tabs[0]:
            allt = matched.copy()
            is_unm = allt['_matched_via'] == 'unmatched'
            matched_low = (~is_unm) & (allt['_usage_total'] <= threshold)
            has_use = (~is_unm) & (allt['_usage_total'] > threshold)
            zero_label = f'Zero/low-use (\u2264 {threshold:g})'
            if treat_unmatched_as_zero:
                allt['_use_status'] = np.where(has_use, 'Has use', zero_label)
                n_zero = int((~has_use).sum())   # matched-low + unmatched
                n_unm = 0
            else:
                allt['_use_status'] = np.where(
                    is_unm, 'Unmatched',
                    np.where(matched_low, zero_label, 'Has use'))
                n_zero = int(matched_low.sum())
                n_unm = int(is_unm.sum())
            n_has_use = int(has_use.sum())

            st.markdown(
                f"**{len(allt):,} titles** \u2014 every holdings title with an explicit "
                "use value (titles with no recorded use carry **0**). This is the "
                "synced file to feed Use Analysis (Coverage-vs-Use) or COUNTER triage."
            )
            if treat_unmatched_as_zero:
                st.caption(
                    f"\U0001F4CB {n_zero:,} zero/low-use (incl. unmatched) \u00b7 "
                    f"{n_has_use:,} with use. Sort or filter on **Use Status**."
                )
            else:
                st.caption(
                    f"\U0001F4CB {n_zero:,} zero/low-use \u00b7 {n_unm:,} unmatched "
                    "(inferred 0 \u2014 could be a usage-report gap) \u00b7 "
                    f"{n_has_use:,} with use. Sort or filter on **Use Status**."
                )

            comb_cols = [c for c in (list(display_cols)
                         + ['_use_status', '_matched_via', '_usage_total'])
                         if c in allt.columns]
            comb_display = allt[comb_cols].rename(columns={
                '_use_status': 'Use Status',
                '_matched_via': 'Matched Via',
                '_usage_total': 'Total Use',
            }).sort_values('Total Use')
            st.dataframe(comb_display, use_container_width=True, height=500, hide_index=True)

            _all_csv = _annotate_csv(
                comb_display, notes,
                extra_meta={'Tool': 'Zero-Use Identifier',
                            'View': 'All titles (explicit zeros)',
                            'Threshold': threshold,
                            'Treat unmatched as zero-use': treat_unmatched_as_zero,
                            'Holdings rows': total_holdings,
                            'Usage rows': len(usage_df),
                            'Subjects retained': bool(h_subj),
                            'Match keys': ', '.join(shared_keys),
                            'Zero/low-use titles': n_zero,
                            'Unmatched titles': n_unm,
                            'Titles with use': n_has_use}
            )
            st.download_button(
                "\U0001F4E5 All titles, explicit zeros (zipped CSV)",
                _zip_one_csv(_all_csv, "all_titles_explicit_zeros.csv"),
                "all_titles_explicit_zeros.zip", "application/zip",
                key="zu_dl_all")
            st.caption("Downloads as a zipped CSV \u2014 unzip to open the CSV in Excel.")

        # --- Optional output: Zero-use-only list (opt-in) ---
        if also_zero_list:
            with out_tabs[1]:
                if treat_unmatched_as_zero:
                    st.markdown(
                        f"**{len(zero_use):,} titles** flagged as zero/low-use "
                        f"(matched \u2264 {threshold:g} use **plus** unmatched "
                        f"titles){cutoff_msg}."
                    )
                    st.caption(
                        "\U0001F4CC Unmatched titles are folded in because **Treat "
                        "unmatched as zero-use** is on. The `Matched Via` column "
                        "shows which group each title came from."
                    )
                else:
                    st.markdown(
                        f"**{len(zero_use):,} titles** in your holdings have \u2264 "
                        f"{threshold:g} recorded use{cutoff_msg}."
                    )
                    if n_unmatched > 0:
                        st.caption(
                            f"\U0001F4CC This list excludes the **{n_unmatched:,} "
                            "unmatched** titles (they appear in the master, tagged "
                            "`Unmatched`). To treat them as zero-use here, turn on "
                            "**Treat unmatched as zero-use** in the sidebar."
                        )
                zero_cols = [c for c in (list(display_cols) + ['_matched_via', '_usage_total'])
                             if c in zero_use.columns]
                display_df = zero_use[zero_cols].rename(columns={
                    '_matched_via': 'Matched Via',
                    '_usage_total': 'Total Use',
                }).sort_values('Total Use')
                st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)

                _zu_csv = _annotate_csv(
                    display_df, notes,
                    extra_meta={'Tool': 'Zero-Use Identifier',
                                'View': 'Zero-use titles'
                                        + (' (incl. unmatched)' if treat_unmatched_as_zero else ''),
                                'Threshold': threshold,
                                'Pub-year cutoff': pubyear_cutoff or 'none',
                                'Treat unmatched as zero-use': treat_unmatched_as_zero,
                                'Holdings rows': total_holdings,
                                'Usage rows': len(usage_df),
                                'Subjects retained': bool(h_subj),
                                'Match keys': ', '.join(shared_keys)}
                )
                st.download_button(
                    "\U0001F4E5 Zero-use titles (zipped CSV)",
                    _zip_one_csv(_zu_csv, "zero_use_titles.csv"),
                    "zero_use_titles.zip", "application/zip",
                    key="zu_dl_zero")
                st.caption("Downloads as a zipped CSV \u2014 unzip to open the CSV in Excel.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Check that both CSVs have at least a Title column or a shared "
                "identifier (ISBN/ISSN/DOI/OCLC).")


# =====================================================================
# =====================================================================
# TOOL 4: OVERLAP & UNIQUENESS  (e-journal package overlap analyzer)
# =====================================================================
# "What's unique to each database — and what would we lose by cancelling it?"
#
# Reads an Alma-style electronic-journal coverage / A-to-Z export where each
# row is a Title x Electronic Collection (x Interface) with a coverage
# statement. For any chosen database it classifies every title into:
#
#   • Sole source     — the title exists in NO other database in the file.
#                       Cancelling = you lose the title outright.
#   • Unique coverage — the title is held elsewhere too, but THIS database is
#                       the only source for some span of years. Cancelling =
#                       you open a coverage gap.
#   • Redundant       — every year this database provides is also provided by
#                       at least one other database. Cancelling = no loss.
#
# The coverage math (interval union + subtraction at day resolution) is what
# makes the "consider coverage" part work: a title can be redundant by name
# but irreplaceable by date range, and this tool tells them apart.
# =====================================================================
# =====================================================================


def _ovl_parse_one_date(s, is_end):
    """Parse a single coverage date token (YYYY, YYYY-M, or YYYY-M-D) into a
    date. Year-only / month-only tokens expand to the start of the span for a
    start date and the end of the span for an end date, so "1847 until 1847"
    means the whole of 1847."""
    from datetime import date, timedelta
    parts = str(s).strip().split('-')
    try:
        y = int(parts[0])
    except (ValueError, IndexError):
        return None
    if y < 1 or y > 2999:
        return None
    if len(parts) == 1:
        return date(y, 12, 31) if is_end else date(y, 1, 1)
    try:
        m = int(parts[1])
    except ValueError:
        return date(y, 12, 31) if is_end else date(y, 1, 1)
    m = min(max(m, 1), 12)
    if len(parts) == 2:
        if is_end:
            nxt = date(y + (m == 12), (m % 12) + 1, 1)
            return nxt - timedelta(days=1)
        return date(y, m, 1)
    try:
        d = int(parts[2])
        return date(y, m, d)
    except (ValueError, IndexError):
        # Bad day-of-month — fall back to month bounds
        if is_end:
            nxt = date(y + (m == 12), (m % 12) + 1, 1)
            return nxt - timedelta(days=1)
        return date(y, m, 1)


def _ovl_parse_coverage(text, present):
    """Parse a coverage statement into a list of (start_date, end_date, ongoing)
    intervals. Handles multiple "Available from X until Y;" clauses in one cell.
    An open-ended clause (no "until") ends at `present` and is flagged ongoing."""
    if text is None or (isinstance(text, float)):
        return []
    s = str(text)
    if not s.strip() or s.lower() == 'nan':
        return []
    intervals = []
    for mm in re.finditer(r'from\s+([\d\-]+)(?:\s+until\s+([\d\-]+))?', s, re.I):
        start = _ovl_parse_one_date(mm.group(1), is_end=False)
        if start is None:
            continue
        if mm.group(2):
            end = _ovl_parse_one_date(mm.group(2), is_end=True)
            ongoing = False
        else:
            end = present
            ongoing = True
        if end is not None and end >= start:
            intervals.append((start, end, ongoing))
    return intervals


def _ovl_merge(intervals):
    """Merge a list of (start, end) intervals into non-overlapping, sorted pieces."""
    from datetime import timedelta
    segs = sorted(intervals)
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _ovl_subtract(target, others):
    """Return the pieces of `target` not covered by the union of `others`.
    Both are lists of (start, end) tuples. Target is merged first so a title
    listed more than once in the same database isn't double-counted. Result is
    a list of non-overlapping (start, end) pieces."""
    from datetime import timedelta
    merged = _ovl_merge(others)
    target = _ovl_merge(target)
    result = []
    for s, e in target:
        cur = s
        for ms, me in merged:
            if me < cur or ms > e:
                continue
            if ms > cur:
                result.append((cur, min(ms - timedelta(days=1), e)))
            cur = max(cur, me + timedelta(days=1))
            if cur > e:
                break
        if cur <= e:
            result.append((cur, e))
    return [(s, e) for s, e in result if e >= s]


def _ovl_span_years(ivs):
    """Total span of a list of (start, end) intervals, in fractional years."""
    return sum((e - s).days + 1 for s, e in ivs) / 365.25


def _ovl_fmt_ranges(ivs):
    """Render intervals as a compact human-readable string like '1925–1985; 2014–2015'."""
    out = []
    for s, e in ivs:
        if s.year == e.year:
            out.append(f"{s.year}")
        else:
            out.append(f"{s.year}\u2013{e.year}")
    return "; ".join(out)


def _ovl_classify(df, group_col, title_key_col, title_disp_col, coverage_col,
                  min_years):
    """Classify every (database, title) pair. Returns a long DataFrame with one
    row per database-title combination.

    Columns: database, title, status, unique_years, unique_ranges,
             other_count, also_in.
    """
    from datetime import date

    present = date.today()

    # Parse coverage once per row.
    parsed = df[coverage_col].apply(lambda t: _ovl_parse_coverage(t, present))

    # Build title -> list of (group, [intervals]) so each title is touched once.
    title_recs = defaultdict(list)
    title_disp = {}
    for grp, tkey, tdisp, ivs in zip(df[group_col], df[title_key_col],
                                     df[title_disp_col], parsed):
        if tkey is None or (isinstance(tkey, float) and pd.isna(tkey)):
            continue
        tkey = str(tkey).strip()
        if not tkey:
            continue
        if pd.isna(grp) or not str(grp).strip():
            continue
        title_recs[tkey].append((str(grp).strip(), ivs))
        title_disp.setdefault(tkey, tdisp)

    rows = []
    for tkey, recs in title_recs.items():
        groups_here = {g for g, _ in recs}
        disp = title_disp.get(tkey, tkey)
        for g in groups_here:
            target = [(s, e) for rg, ivs in recs if rg == g for (s, e, _) in ivs]
            other_groups = sorted({rg for rg, _ in recs if rg != g})
            other = [(s, e) for rg, ivs in recs if rg != g for (s, e, _) in ivs]
            if not other_groups:
                status = "Sole source"
                unique = _ovl_merge(target)
            else:
                unique = _ovl_subtract(target, other)
                yrs = _ovl_span_years(unique)
                status = "Unique coverage" if yrs > min_years else "Redundant"
            rows.append({
                "database": g,
                "title": disp,
                "status": status,
                "unique_years": round(_ovl_span_years(unique), 2),
                "unique_ranges": _ovl_fmt_ranges(unique),
                "other_count": len(other_groups),
                "also_in": ", ".join(other_groups),
            })
    return pd.DataFrame(rows)


# Status display order + colors (Tulane palette + neutral).
_OVL_STATUS_ORDER = ["Sole source", "Unique coverage", "Redundant"]
_OVL_STATUS_COLORS = {
    "Sole source": "#285C4D",      # Tulane green — most irreplaceable
    "Unique coverage": "#71C5E8",  # Tulane blue — partial loss if cancelled
    "Redundant": "#C9CCCE",        # neutral gray — safe to cancel
}


def _ovl_cached_classification(tool_key, uploaded_file, group_col,
                               title_key_col, title_disp_col, coverage_col,
                               min_years, df):
    """Memoize the (somewhat expensive) classification in session_state, keyed
    on file + the settings that affect the result. Returns the long DataFrame."""
    file_key = _make_file_key(uploaded_file)
    sig = (file_key, group_col, title_key_col, coverage_col, min_years)
    slot = st.session_state.get(f"_ovl_cache_{tool_key}")
    if slot and slot.get("sig") == sig:
        return slot["result"]
    result = _ovl_classify(df, group_col, title_key_col, title_disp_col,
                           coverage_col, min_years)
    st.session_state[f"_ovl_cache_{tool_key}"] = {"sig": sig, "result": result}
    return result


def page_overlap_analyzer():
    """Tool 4: Overlap & Uniqueness — which titles are unique to a database,
    accounting for date coverage."""
    TOOL = "overlap"
    st.header("\U0001F9E9 Overlap & Uniqueness")
    st.markdown(
        "**What's unique to each database \u2014 and what would we lose by "
        "cancelling it?** Upload an electronic-journal coverage / A-to-Z export "
        "(one row per title \u00d7 database, with a coverage statement) and this "
        "tool sorts every title into *sole source*, *unique coverage*, or "
        "*redundant* \u2014 taking the actual date ranges into account, not just "
        "the title name."
    )

    with st.expander("\u2139\ufe0f When to use this tool"):
        st.markdown(
            "- **Cancellation review:** Before dropping a package, see exactly "
            "which titles you'd lose outright (*sole source*) and which you'd "
            "keep but with a coverage gap (*unique coverage*).\n"
            "- **Overlap / duplication audit:** Find titles you're paying for in "
            "several databases at once (*redundant*) \u2014 candidates for "
            "trimming without losing any content.\n"
            "- **Big-deal evaluation:** Rank databases by how much irreplaceable "
            "content they hold, so the all-or-nothing packages get scrutinized.\n"
            "- **Pairs well with Use Analysis:** uniqueness tells you what's "
            "*replaceable*; usage (COUNTER or other) tells you what's *used*. Cancel "
            "the titles that are both redundant and unused first."
        )

    with st.expander("\U0001F4D6 How the coverage math works", expanded=False):
        st.markdown("""
        Each row is a **title in one database** with a coverage statement like
        *"Available from 1925-7-19 until 1985-12-31"* (an open-ended one with no
        *until* is treated as running to today).

        For a chosen database, each of its titles is compared against **every
        other database in the file**:

        1. **Sole source** \u2014 the title appears in no other database. You'd
           lose it entirely.
        2. **Unique coverage** \u2014 the title is elsewhere too, but this
           database covers a date span *no other database covers*. The years it
           uniquely provides are listed so you can judge the gap.
        3. **Redundant** \u2014 every year this database provides is already
           provided elsewhere. Safe to drop on coverage grounds.

        The comparison is done by **interval subtraction at day resolution**:
        the tool builds the union of all other databases' date ranges for that
        title, then checks whether this database's range pokes outside it.

        **Caveat \u2014 metadata granularity:** coverage stated as a bare year
        (e.g. *1847*) is treated as the whole year. If one source gives precise
        dates and another only a year, you can see small *unique coverage* spans
        that are really just rounding. The **materiality threshold** below lets
        you ignore gaps under N years.
        """)

    uploaded_file = st.file_uploader(
        "Upload coverage / A-to-Z export (CSV or Excel)",
        type=['csv', 'xls', 'xlsx'], key="ovl_upload",
        help="One row per title \u00d7 database, with a coverage statement "
             "column. Alma 'Electronic Journal Coverage' exports work as-is."
    )

    if not uploaded_file:
        st.info("Upload a coverage export to begin. It needs a **title** column, "
                "a **coverage** column, and a **database/collection** (or "
                "**interface**) column.")
        return

    try:
        cached = _cached_df_for_tool(TOOL, uploaded_file)
        if cached is not None:
            df = cached.copy()
        else:
            df = _load_csv_chunked(uploaded_file.getvalue(), uploaded_file.name)
            _store_cached_df(TOOL, uploaded_file, df)

        st.success(f"\u2705 Loaded **{len(df):,}** rows.")

        # ---- Column detection ----
        coverage_col = find_column(df, COVERAGE_ALIASES)
        collection_col = find_column(df, COLLECTION_ALIASES)
        interface_col = find_column(df, INTERFACE_ALIASES)
        title_disp_col = find_column(df, TITLE_ALIASES)
        title_norm_col = find_column(df, NORM_TITLE_ALIASES)

        # Which grouping dimensions are available?
        dim_options = []
        if collection_col:
            dim_options.append("Database / collection")
        if interface_col:
            dim_options.append("Interface / provider")

        with st.expander("\U0001F50D Column detection & overrides", expanded=False):
            st.caption(
                f"Title: `{title_disp_col}` \u00b7 Normalized title: "
                f"`{title_norm_col}` \u00b7 Coverage: `{coverage_col}` \u00b7 "
                f"Collection: `{collection_col}` \u00b7 Interface: `{interface_col}`"
            )
            cols = list(df.columns)

            def _idx(col):
                return cols.index(col) + 1 if col in cols else 0

            opts = ["\u2014 none \u2014"] + cols
            title_disp_col = st.selectbox(
                "Title column (display)", opts, index=_idx(title_disp_col),
                key="ovl_title_override")
            title_disp_col = None if title_disp_col == "\u2014 none \u2014" else title_disp_col

            norm_pick = st.selectbox(
                "Normalized-title column (for matching; optional)",
                opts, index=_idx(title_norm_col), key="ovl_norm_override")
            title_norm_col = None if norm_pick == "\u2014 none \u2014" else norm_pick

            cov_pick = st.selectbox(
                "Coverage column", opts, index=_idx(coverage_col),
                key="ovl_cov_override")
            coverage_col = None if cov_pick == "\u2014 none \u2014" else cov_pick

            coll_pick = st.selectbox(
                "Database / collection column", opts, index=_idx(collection_col),
                key="ovl_coll_override")
            collection_col = None if coll_pick == "\u2014 none \u2014" else coll_pick

            iface_pick = st.selectbox(
                "Interface / provider column", opts, index=_idx(interface_col),
                key="ovl_iface_override")
            interface_col = None if iface_pick == "\u2014 none \u2014" else iface_pick

        # Rebuild dimension options after any overrides.
        dim_options = []
        if collection_col:
            dim_options.append("Database / collection")
        if interface_col:
            dim_options.append("Interface / provider")

        # ---- Validation ----
        if not title_disp_col:
            st.error("\u274C Need a **title** column. Set one in the overrides above.")
            return
        if not coverage_col:
            st.error("\u274C Need a **coverage** column (e.g. *Coverage "
                     "Information Combined*). Set one in the overrides above.")
            return
        if not dim_options:
            st.error("\u274C Need a **database/collection** or **interface** "
                     "column to compare across. Set one in the overrides above.")
            return

        # ---- Settings ----
        sc1, sc2 = st.columns([2, 3])
        with sc1:
            dim_choice = st.radio("Compare uniqueness by:", dim_options,
                                  index=0, key="ovl_dim")
            group_col = (collection_col if dim_choice == "Database / collection"
                         else interface_col)
        with sc2:
            min_years = st.slider(
                "Materiality threshold \u2014 minimum unique span to count as "
                "*unique coverage* (years)",
                0.0, 5.0, 0.0, 0.25, key="ovl_minyears",
                help="A title held elsewhere counts as 'unique coverage' only if "
                     "this database uniquely provides at least this many years. "
                     "Raise it to ignore tiny gaps from year-level metadata. "
                     "0 = flag any unique span.")

        # ---- Title matching key ----
        # Prefer a normalized-title column; otherwise normalize the display title.
        df = df.copy()
        if title_norm_col and title_norm_col in df.columns:
            df["_ovl_key"] = df[title_norm_col].apply(
                lambda v: normalize_text(v) if pd.notna(v) and str(v).strip()
                else None)
            # Fall back to display title where the normalized cell is blank.
            blank = df["_ovl_key"].isna() | (df["_ovl_key"] == "")
            df.loc[blank, "_ovl_key"] = df.loc[blank, title_disp_col].apply(
                lambda v: normalize_text(v) if pd.notna(v) else None)
        else:
            df["_ovl_key"] = df[title_disp_col].apply(
                lambda v: normalize_text(v) if pd.notna(v) else None)

        # ---- Classify (memoized) ----
        with st.spinner("Comparing coverage across databases\u2026"):
            long_df = _ovl_cached_classification(
                TOOL, uploaded_file, group_col, "_ovl_key", title_disp_col,
                coverage_col, min_years, df)

        if long_df.empty:
            st.warning("No title/database pairs could be built. Check that the "
                       "title and coverage columns are mapped correctly.")
            return

        notes = _notes_widget(
            TOOL,
            placeholder="e.g., July cancellation review. America's Historical "
                        "Newspapers holds 'Advocate' 1925\u20131985 \u2014 "
                        "no replacement; flag for retention.")
        _reset_tray(TOOL)

        n_databases = long_df["database"].nunique()
        n_titles = long_df["title"].nunique()
        meta = {"Compared by": dim_choice, "Group column": group_col,
                "Materiality threshold (yrs)": min_years,
                "Databases": n_databases, "Distinct titles": n_titles}

        st.markdown("---")
        tab_profile, tab_drill, tab_lookup = st.tabs([
            "\U0001F4CA Uniqueness by database",
            "\U0001F50E Drill into one database",
            "\U0001F50D Title lookup",
        ])

        # =============================================================
        # TAB 1 — Uniqueness profile across all databases
        # =============================================================
        with tab_profile:
            prof = (long_df.groupby("database")
                    .agg(Titles=("title", "count"),
                         Sole_source=("status",
                                      lambda s: int((s == "Sole source").sum())),
                         Unique_coverage=("status",
                                          lambda s: int((s == "Unique coverage").sum())),
                         Redundant=("status",
                                    lambda s: int((s == "Redundant").sum())))
                    .reset_index())
            prof["Irreplaceable"] = prof["Sole_source"] + prof["Unique_coverage"]
            prof["% redundant"] = (prof["Redundant"] / prof["Titles"] * 100).round(0)
            prof = prof.sort_values(["Irreplaceable", "Titles"],
                                    ascending=False).reset_index(drop=True)

            total_redundant = int((long_df["status"] == "Redundant").sum())
            total_pairs = len(long_df)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Databases", f"{n_databases:,}")
            k2.metric("Distinct titles", f"{n_titles:,}")
            k3.metric("Title placements", f"{total_pairs:,}",
                      help="Title \u00d7 database combinations across the file.")
            k4.metric("Redundant placements",
                      f"{total_redundant/max(1,total_pairs)*100:.0f}%",
                      help="Share of placements whose coverage is fully duplicated "
                           "elsewhere \u2014 the raw overlap in the file.")

            st.markdown("##### How irreplaceable is each database?")
            st.caption("Sorted by *irreplaceable* (sole source + unique coverage). "
                       "A database that's mostly redundant is the safest to cancel; "
                       "one with many sole-source titles is the costliest to lose.")

            display_prof = prof.rename(columns={
                "database": "Database", "Sole_source": "Sole source",
                "Unique_coverage": "Unique coverage"})
            st.dataframe(
                display_prof[["Database", "Titles", "Sole source",
                              "Unique coverage", "Redundant", "Irreplaceable",
                              "% redundant"]],
                use_container_width=True, hide_index=True)

            # Stacked bar — top N databases by title count.
            top_n = min(20, len(prof))
            bar_df = prof.head(top_n).iloc[::-1]  # reverse so largest at top
            fig = go.Figure()
            for status, col in [("Sole_source", "Sole source"),
                                ("Unique_coverage", "Unique coverage"),
                                ("Redundant", "Redundant")]:
                fig.add_trace(go.Bar(
                    y=bar_df["database"], x=bar_df[status], name=col,
                    orientation="h",
                    marker_color=_OVL_STATUS_COLORS[col],
                    hovertemplate="%{y}<br>" + col + ": %{x}<extra></extra>"))
            fig.update_layout(
                barmode="stack", height=max(320, 26 * top_n + 120),
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0),
                xaxis_title="Titles", yaxis_title=None,
                font=dict(family="DM Sans, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)

            _decision_box(
                "Reading this",
                "- **Lots of green (sole source)** \u2192 irreplaceable content; "
                "cancelling means buying the titles back elsewhere or losing them.\n"
                "- **Lots of blue (unique coverage)** \u2192 you'd keep the titles "
                "but open date gaps; check the drill-down for which years.\n"
                "- **Mostly gray (redundant)** \u2192 the database duplicates "
                "coverage you already have \u2014 the cleanest cancellation "
                "candidate on content grounds (still check usage in COUNTER).")

            csv = _annotate_csv(display_prof, notes, extra_meta=meta)
            _dl("\U0001F4E5 Uniqueness profile (CSV)", csv,
                "uniqueness_profile.csv", "text/csv",
                key="ovl_dl_profile", tool_key=TOOL)

        # =============================================================
        # TAB 2 — Drill into one database
        # =============================================================
        with tab_drill:
            db_list = sorted(long_df["database"].unique())
            pick = st.selectbox("Database to examine", db_list, key="ovl_pick_db")
            sub = long_df[long_df["database"] == pick].copy()

            n_sole = int((sub["status"] == "Sole source").sum())
            n_uniq = int((sub["status"] == "Unique coverage").sum())
            n_red = int((sub["status"] == "Redundant").sum())

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Titles", f"{len(sub):,}")
            d2.metric("Sole source", f"{n_sole:,}",
                      help="Lost entirely if cancelled.")
            d3.metric("Unique coverage", f"{n_uniq:,}",
                      help="Kept, but with a date gap if cancelled.")
            d4.metric("Redundant", f"{n_red:,}",
                      help="Fully duplicated elsewhere.")

            irreplaceable = n_sole + n_uniq
            if irreplaceable == 0:
                st.success(
                    f"\u2705 **Every title in *{pick}* is fully covered by other "
                    f"databases in this file.** On coverage grounds, cancelling it "
                    f"loses no content. (Confirm usage and any non-journal content "
                    f"separately.)")
            else:
                st.warning(
                    f"\u26A0\ufe0f Cancelling **{pick}** would lose **{n_sole}** "
                    f"title(s) outright and open coverage gaps on **{n_uniq}** "
                    f"more. **{n_red}** of its {len(sub)} titles are safely "
                    f"duplicated elsewhere.")

            status_filter = st.radio(
                "Show", ["All", "Sole source", "Unique coverage", "Redundant"],
                horizontal=True, key="ovl_status_filter")
            view = sub if status_filter == "All" else sub[sub["status"] == status_filter]

            # Order: sole source first, then unique coverage (by span desc), then redundant.
            order_map = {"Sole source": 0, "Unique coverage": 1, "Redundant": 2}
            view = view.assign(_o=view["status"].map(order_map)).sort_values(
                ["_o", "unique_years"], ascending=[True, False]).drop(columns="_o")

            display = view.rename(columns={
                "title": "Title", "status": "Status",
                "unique_years": "Unique years",
                "unique_ranges": "Unique coverage (years)",
                "also_in": "Also available in"})
            st.dataframe(
                display[["Title", "Status", "Unique years",
                         "Unique coverage (years)", "Also available in"]],
                use_container_width=True, hide_index=True)

            safe_name = re.sub(r'[^\w\-]+', '_', pick)[:60].strip('_') or "database"
            csv = _annotate_csv(
                display[["Title", "Status", "Unique years",
                         "Unique coverage (years)", "Also available in"]],
                notes, extra_meta={**meta, "Database examined": pick})
            _dl(f"\U0001F4E5 {pick[:40]} \u2014 title classification (CSV)", csv,
                f"uniqueness_{safe_name}.csv", "text/csv",
                key="ovl_dl_drill", tool_key=TOOL)

        # =============================================================
        # TAB 3 — Title lookup
        # =============================================================
        with tab_lookup:
            st.caption("Search a title to see every database that holds it, the "
                       "coverage each provides, and where the unique years sit.")
            q = st.text_input("Title contains", key="ovl_lookup_q",
                              placeholder="e.g., Advocate")
            if q and q.strip():
                ql = q.strip().lower()
                hits = long_df[long_df["title"].str.lower().str.contains(
                    re.escape(ql), na=False)]
                titles_found = sorted(hits["title"].unique())
                if not titles_found:
                    st.info("No titles match that search.")
                else:
                    if len(titles_found) > 1:
                        chosen = st.selectbox(
                            f"{len(titles_found)} matching titles",
                            titles_found, key="ovl_lookup_pick")
                    else:
                        chosen = titles_found[0]
                    rows = long_df[long_df["title"] == chosen].copy()
                    order_map = {"Sole source": 0, "Unique coverage": 1, "Redundant": 2}
                    rows = rows.assign(_o=rows["status"].map(order_map)).sort_values(
                        ["_o", "unique_years"], ascending=[True, False]).drop(columns="_o")
                    n_db = rows["database"].nunique()
                    st.markdown(f"**{chosen}** \u2014 held in **{n_db}** "
                                f"database{'s' if n_db != 1 else ''}.")
                    disp = rows.rename(columns={
                        "database": "Database", "status": "Status in this database",
                        "unique_years": "Unique years",
                        "unique_ranges": "Unique coverage (years)"})
                    st.dataframe(
                        disp[["Database", "Status in this database",
                              "Unique years", "Unique coverage (years)"]],
                        use_container_width=True, hide_index=True)
                    if (rows["status"] == "Redundant").all():
                        st.caption("Every copy of this title is duplicated \u2014 "
                                   "no single database is the sole source for any year.")

        st.markdown("---")
        _render_download_tray(TOOL, zip_filename="overlap_uniqueness.zip")

    except Exception as e:
        st.error(f"\u274C Error: {e}")
        st.info("Check that the file has a title column, a coverage column, and "
                "a database/collection (or interface) column. Alma 'Electronic "
                "Journal Coverage' exports work without changes.")


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

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="tool-card">
            <h3>🗺️ Collection Profiler</h3>
            <p><em>What does our collection look like?</em></p>
            <p>Structure only — LC sunburst, treemap, subject word cloud, and
            sub-class range analysis. Map disciplinary strengths across 1M+ records
            via LC Analysis and Subject Term Analysis. Usage-driven views live in
            Use Analysis.</p>
            <hr>
            <p><strong>Use for:</strong></p>
            <ul>
                <li>Baseline & accreditation reports</li>
                <li>Liaison prep & subject policy revision</li>
                <li>Budget justifications</li>
                <li>Holdings distribution by LC & subject</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="tool-card">
            <h3>📈 Use Analysis</h3>
            <p><em>What's getting used — and is it worth keeping?</em></p>
            <p>One tool for all usage-driven analysis. Print circulation
            (subject + usage), formal COUNTER 5 reports, or any other usage data.
            Coverage vs. Use, top titles, gap-vs-use, cost-per-use, monthly trends,
            and dead-weight titles. Feed it the explicit-zero master from the
            Zero-Use Identifier.</p>
            <hr>
            <p><strong>Use for:</strong></p>
            <ul>
                <li>Coverage vs. Use (print or e-resource Branch A)</li>
                <li>Database renewals & cancellation review</li>
                <li>Big Deal evaluation, monthly trends</li>
                <li>Print weeding by low circulation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Zero-Use sits on its own row (wider, since it doesn't have a partner card)
    st.markdown("""
    <div class="tool-card">
        <h3>🔍 Zero-Use Identifier</h3>
        <p><em>What do we own that isn't being used?</em></p>
        <p>Compare a holdings list against a usage report to surface
        titles, journals, or databases with no use at all.</p>
        <hr>
        <p><strong>Use for:</strong></p>
        <ul>
            <li>E-journal & database cancellation prep</li>
            <li>Off-site storage candidates</li>
            <li>Dead-weight in big-deal packages</li>
            <li>Renewal evidence for admin/faculty</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-card">
        <h3>🧩 Overlap & Uniqueness</h3>
        <p><em>What's unique to each database — and what would we lose by cancelling it?</em></p>
        <p>Read an e-journal coverage / A-to-Z export and classify every title
        per database as <strong>sole source</strong>, <strong>unique
        coverage</strong>, or <strong>redundant</strong> — accounting for the
        actual date ranges, not just the title name.</p>
        <hr>
        <p><strong>Use for:</strong></p>
        <ul>
            <li>Package cancellation impact (what's truly lost)</li>
            <li>Overlap / duplication audits across databases</li>
            <li>Coverage-gap analysis before dropping a source</li>
            <li>Ranking databases by irreplaceable content</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick decision guide")
    st.markdown("""
    | You need to… | Use |
    |---|---|
    | Show what the collection covers (or doesn't) | **Collection Profiler** → LC Analysis |
    | Map disciplinary strengths via subject terms | **Collection Profiler** → Subject Term Analysis |
    | See holdings distribution by LC sub-class range | **Collection Profiler** |
    | Pick books to weed by low circulation | **Use Analysis** → Print circulation |
    | Analyze print circulation against subject + LC | **Use Analysis** → Print circulation |
    | Decide which databases to renew or cancel | **Use Analysis** → COUNTER 5 |
    | Run monthly-trend / cost-per-use on COUNTER reports | **Use Analysis** → COUNTER 5 |
    | Analyze any other (non-COUNTER) usage export | **Use Analysis** → Other usage data |
    | Find areas with strong use relative to holdings (or weak) | **Use Analysis** (Coverage vs. Use) |
    | Find what you own that's never been used | **Zero-Use Identifier** |
    | Identify e-journal/package titles with no use | **Zero-Use Identifier** (holdings vs. COUNTER) |
    | See which titles are unique to a database (by coverage) | **Overlap & Uniqueness** |
    | Estimate what content a package cancellation would lose | **Overlap & Uniqueness** |
    | Find titles duplicated across several databases | **Overlap & Uniqueness** (Redundant) |
    """)

    st.markdown("---")
    with st.expander("ℹ️ About this dashboard"):
        st.markdown("""
        **Version 2.7 (slim)** — four collection-analysis tools:
        - **Collection Profiler** — holdings structure only (LC, Subject Term,
          sub-class ranges, distribution). Usage views moved to Use Analysis.
        - **Use Analysis** — all usage-driven work in one place: print
          circulation (subject + usage), formal COUNTER 5 reports, and other
          usage data. Owns Coverage vs. Use, cost-per-use, monthly trends, and
          dead-weight triage.
        - **Zero-Use Identifier** — holdings vs. usage matching; emits the
          explicit-zero master that feeds Use Analysis.
        - **Overlap & Uniqueness** — e-journal coverage overlap; classifies
          titles per database as sole source / unique coverage / redundant.

        The Acquisition Recommendation Scorer ("what should we buy next?") has
        been extracted into its own standalone app — see `recommender_app.py`.

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
             "📈 Use Analysis",
             "🔍 Zero-Use Identifier",
             "🧩 Overlap & Uniqueness"],
            index=0,
            key="nav"
        )
        st.markdown("---")

    if page == "🏠 Home":
        page_home()
    elif page == "🗺️ Collection Profiler":
        page_collection_profiler()
    elif page == "📈 Use Analysis":
        page_use_analysis()
    elif page == "🔍 Zero-Use Identifier":
        page_zero_use_identifier()
    elif page == "🧩 Overlap & Uniqueness":
        page_overlap_analyzer()

    _footer()


if __name__ == "__main__":
    main()
