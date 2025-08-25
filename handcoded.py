import streamlit as st
import pandas as pd
import os
import uuid

st.set_page_config(page_title="Legal Text Annotator", layout="wide")

# === Constants ===
DATA_FILE = "acts_cleanish.csv"
SAVE_FILE = "validation.csv"
TEXT_COLUMN = "act_raw_text"
ID_COLUMN = "CELEX"

classification_fields = [
    "Definition",
    "Actor", "Actor type",
    "Delegation Binary", "Delegation Type",
    "Derogation Binary", "Derogation Details",
    "Dilution Binary", "Dilution Details",
    "Instrument","Instrument Type",
    "Domain",
    "Target", "Target Type"
]

delegation_types = [
    "None", "Major Operational Delegation", "Minor Procedural Delegation",
    "Joint Delegation", "Self-Execution"
]

yes_no = ["No", "Yes"]

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df[[ID_COLUMN, TEXT_COLUMN]].dropna().drop_duplicates()
    return df.reset_index(drop=True)

data = load_data()

# === App State ===
st.title("⚖️ Legal Annotation Tool")

# Track session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []

def save_current_annotations():
    if st.session_state.annotations:
        df = pd.DataFrame(st.session_state.annotations)
        if os.path.exists(SAVE_FILE):
            df_existing = pd.read_csv(SAVE_FILE)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(SAVE_FILE, index=False)
        st.success(f"Saved {len(st.session_state.annotations)} entries.")
        st.session_state.annotations = []

# === Navigation ===
st.sidebar.title("Navigation")
st.sidebar.write(f"Current document: {st.session_state.index + 1} / {len(data)}")
if st.sidebar.button("⏮ Previous") and st.session_state.index > 0:
    st.session_state.index -= 1
if st.sidebar.button("Next ⏭") and st.session_state.index < len(data) - 1:
    st.session_state.index += 1
if st.sidebar.button("💾 Save to CSV"):
    save_current_annotations()

# === Display Legal Text ===
row = data.iloc[st.session_state.index]
st.subheader(f"CELEX: {row[ID_COLUMN]}")
st.markdown("### Law Text")
st.text_area("Full Text", row[TEXT_COLUMN], height=400, disabled=True)
st.text_area("Core Sentence", "Core sentence", height=200, disabled=True)

# === Annotation Section ===
st.markdown("### Add Core Operational Sentences")
with st.form("annotation_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        article_number = st.text_input("Article Number")
        definition = st.selectbox("Definition", yes_no)
        actor = st.text_input("Actor")
        actor_type = st.text_input("Actor type")
        delegation_binary = st.selectbox("Delegation Binary", yes_no)
        delegation_type = st.selectbox("Delegation Type", delegation_types)
    with col2:
        derogation_binary = st.selectbox("Derogation Binary", yes_no)
        derogation_details = st.text_input("Derogation Details")
        dilution_binary = st.selectbox("Dilution Binary", yes_no)
        dilution_details = st.text_input("Dilution Details")
        instrument = st.text_input("Instrument")
        instrument_type = st.text_input("Instrument Type")
        domain = st.text_input("Domain")
        target = st.text_input("Target")
        target_type = st.text_input("Target Type")

    submitted = st.form_submit_button("➕ Add Sentence")
    if submitted:
        entry = {
            "celex_number": row[ID_COLUMN],
            "Article Number": article_number,
            "Definition": definition,
            "Actor": actor,
            "Actor type": actor_type,
            "Delegation Binary": delegation_binary,
            "Delegation Type": delegation_type,
            "Derogation Binary": derogation_binary,
            "Derogation Details": derogation_details,
            "Dilution Binary": dilution_binary,
            "Dilution Details": dilution_details,
            "Instrument": instrument,
            "Instrument Type": instrument_type,
            "Domain": domain,
            "Target": target,
            "Target Type": target_type
        }
        st.session_state.annotations.append(entry)
        st.success("✅ Added to current session (not yet saved).")

# === Current Session Display ===
if st.session_state.annotations:
    st.markdown("### Current Unsaved Annotations")
    st.dataframe(pd.DataFrame(st.session_state.annotations))

