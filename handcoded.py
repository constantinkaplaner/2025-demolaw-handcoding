import streamlit as st
import pandas as pd
import os
from datetime import datetime
import html
import re  # for safe filename

st.set_page_config(page_title="Legal Text Annotator", layout="wide")

# === Constants (data input is shared; output is per-coder) ===
DATA_FILE = "sample_mapped.csv"

TEXT_COLUMN = "gpt_input_text"
ID_COLUMN = "celex_number"
CORE_SENTENCE = "Core Sentence"
ARTICLE_COLUMN = "Article Number"   # display + mapping
INSTANCE_COL = "__instance_id"      # unique per displayed row

# Labels to VERIFY via checkbox (everything except delegation/derogation/dilution)
FIELDS_TO_VERIFY = [
    "Article Number",
    "Subject",
    "Subject type",
    "Instrument",
    "Instrument Type",
    "Domain",
    "Object",
    "Object Type",
]
# We still track Core Sentence correctness (checkbox on Document tab)
VERIFIED_FIELDS = [CORE_SENTENCE] + FIELDS_TO_VERIFY

delegation_types = [
    "None", "Major Operational Delegation", "Minor Procedural Delegation",
    "Joint Delegation", "Self-Execution"
]
yes_no = ["No", "Yes"]

# Canonical/final columns
BASE_COLUMNS = [
    "coder_id",
    "celex_number",
    "Article Number",
    "Core Sentence",
    "Definition",
    "Subject",
    "Subject type",
    "Delegation Binary",
    "Delegation Type",
    "Derogation Binary",
    "Derogation Details",
    "Dilution Binary",
    "Dilution Details",
    "Instrument",
    "Instrument Type",
    "Domain",
    "Object",
    "Object Type",
]

# Per-field validation audit columns (found + correct)
VERIFICATION_META_COLUMNS = []
for fld in VERIFIED_FIELDS:
    VERIFICATION_META_COLUMNS.extend([f"{fld} (found)", f"{fld} (correct)"])

ALL_COLUMNS = [INSTANCE_COL] + BASE_COLUMNS + VERIFICATION_META_COLUMNS + ["validated_at"]
PRIMARY_KEYS = [INSTANCE_COL]  # one row per instance (per coder file)

# === Helpers ===
def _s(val, default=""):
    if pd.isna(val):
        return default
    return str(val)

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def _normalize_entry_types(entry: dict):
    e = {k: _s(entry.get(k, "")) for k in ALL_COLUMNS}
    if e["Definition"] not in yes_no: e["Definition"] = yes_no[0]
    if e["Delegation Binary"] not in yes_no: e["Delegation Binary"] = yes_no[0]
    if e["Derogation Binary"] not in yes_no: e["Derogation Binary"] = yes_no[0]
    if e["Dilution Binary"] not in yes_no: e["Dilution Binary"] = yes_no[0]
    if e["Delegation Type"] not in delegation_types: e["Delegation Type"] = delegation_types[0]
    for fld in VERIFIED_FIELDS:
        ck = f"{fld} (correct)"
        e[ck] = "Yes" if e.get(ck, "").strip().lower() in {"yes", "true", "1"} else "No"
    return e

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def html_escape(s: str) -> str:
    return html.escape(_s(s))

def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)

# === Styles ===
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { margin-bottom: .25rem; }
.core-box {
  background:#fff; color:#111827; border:1px solid #e5e7eb; border-left:6px solid #6366f1;
  border-radius:12px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,.04);
  max-height: 320px; overflow:auto;
}
.core-title { font-weight:600; margin-bottom:6px; }
.core-body { white-space:pre-wrap; line-height:1.5; font-size:1rem; }
.badge { display:inline-block; padding:.15rem .4rem; border-radius:8px; background:#eef2ff; border:1px solid #dbe4ff; font-size:12px; }

/* Tab callout banners */
.callout {
  margin: 0 0 12px 0;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
  border-left-width: 6px;
  background: #f8fafc;
  font-size: 0.95rem;
}
.callout b { font-weight: 600; }
.callout.info   { border-left-color:#3b82f6; background:#eff6ff; }
.callout.verify { border-left-color:#10b981; background:#ecfdf5; }
.callout.ops    { border-left-color:#8b5cf6; background:#f5f3ff; }
</style>
""", unsafe_allow_html=True)

# === Coder ID gate ===
st.title("⚖️ Legal Annotation Tool")
if "coder_id" not in st.session_state:
    st.session_state["coder_id"] = ""
st.text_input("Enter your insignia (Coder ID):", key="coder_id", placeholder="e.g. AB12")
if not st.session_state.coder_id.strip():
    st.info("Please enter your insignia to start annotating.")
    st.stop()

# === Per-coder SAVE_FILE (separate CSV per coder) ===
SAVE_DIR = "validations"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_FILE = os.path.join(SAVE_DIR, f"validation_{_safe_filename(st.session_state.coder_id)}.csv")

# === Load Data ===
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file: {path}")
    df = pd.read_csv(path)

    needed = [ID_COLUMN, TEXT_COLUMN, CORE_SENTENCE, ARTICLE_COLUMN] + FIELDS_TO_VERIFY
    for c in set(needed):
        if c not in df.columns:
            df[c] = ""

    if ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        missing = [c for c in [ID_COLUMN, TEXT_COLUMN] if c not in df.columns]
        raise KeyError(f"Missing required columns in {path}: {missing}")

    keep = [ID_COLUMN, TEXT_COLUMN, CORE_SENTENCE] + list(dict.fromkeys([ARTICLE_COLUMN] + FIELDS_TO_VERIFY))
    df = df[keep].dropna(subset=[ID_COLUMN, TEXT_COLUMN]).reset_index(drop=True)
    # DO NOT drop_duplicates: every row shown is an instance.
    df[INSTANCE_COL] = df.index.astype(str)  # will pad after load
    return df

def build_instance_map(df: pd.DataFrame, pad_width: int):
    """(celex, article_found, core_found) -> padded instance_id"""
    m = {}
    for i, r in df.iterrows():
        key = (
            _s(r[ID_COLUMN]).strip(),
            _s(r.get(ARTICLE_COLUMN, "")).strip(),
            _s(r.get(CORE_SENTENCE, "")).strip(),
        )
        m[key] = str(i).zfill(pad_width)
    return m

def _dedupe_saved(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # numeric order by instance id + timestamp (latest wins)
    order = pd.to_numeric(df[INSTANCE_COL], errors="coerce")
    if "validated_at" in df.columns:
        ts = pd.to_datetime(df["validated_at"], errors="coerce")
        df = df.assign(_order=order, _ts=ts).sort_values(
            by=["_order", "_ts"], ascending=True
        ).drop(columns=["_order", "_ts"])
    else:
        df = df.assign(_order=order).sort_values(by=["_order"]).drop(columns=["_order"])

    df = df.drop_duplicates(subset=[INSTANCE_COL], keep="last").reset_index(drop=True)
    return df

def migrate_and_filter(df: pd.DataFrame, instance_map: dict, pad_width: int, drop_orphans=True) -> pd.DataFrame:
    """Map legacy rows to current dataset instance ids; drop rows not belonging to current dataset."""
    if df.empty:
        return df

    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    if INSTANCE_COL not in df.columns:
        df[INSTANCE_COL] = ""

    # fallback columns for legacy files
    art_found_col = f"{ARTICLE_COLUMN} (found)" if f"{ARTICLE_COLUMN} (found)" in df.columns else ARTICLE_COLUMN
    core_found_col = f"{CORE_SENTENCE} (found)" if f"{CORE_SENTENCE} (found)" in df.columns else CORE_SENTENCE

    missing_id = df[INSTANCE_COL].astype(str).str.len().eq(0)
    if missing_id.any():
        inst_ids = []
        for _, r in df.loc[missing_id].iterrows():
            key = (
                _s(r.get("celex_number", "")).strip(),
                _s(r.get(art_found_col, "")).strip(),
                _s(r.get(core_found_col, "")).strip(),
            )
            inst_ids.append(instance_map.get(key, ""))  # "" if not in current dataset
        df.loc[missing_id, INSTANCE_COL] = inst_ids

    # pad all instance ids
    df[INSTANCE_COL] = df[INSTANCE_COL].astype(str).str.zfill(pad_width)

    if drop_orphans:
        df = df[df[INSTANCE_COL].astype(str).str.len() > 0].reset_index(drop=True)

    return _dedupe_saved(df)

def load_saved(path: str, instance_map: dict, pad_width: int) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=ALL_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=ALL_COLUMNS)

    # normalize
    def _fix_row(row):
        row = row.to_dict()
        row = _normalize_entry_types(row)
        return pd.Series(row)
    df = df.apply(_fix_row, axis=1)

    # migrate legacy rows -> link to current dataset instances; drop orphans
    df = migrate_and_filter(df, instance_map, pad_width, drop_orphans=True)
    return df.reset_index(drop=True)

# Load data (shared)
try:
    data = load_data(DATA_FILE)
except Exception as e:
    st.exception(e); st.stop()

# Zero-pad instance ids so lexicographic == numeric order
PAD_WIDTH = len(str(max(len(data) - 1, 0)))  # e.g., 25 -> "00".."24"
data[INSTANCE_COL] = data.index.astype(str).str.zfill(PAD_WIDTH)

# Map (celex, article, core) -> padded id for legacy migration
INSTANCE_MAP = build_instance_map(data, PAD_WIDTH)

# Load current coder's saved file only
saved_df = load_saved(SAVE_FILE, INSTANCE_MAP, PAD_WIDTH)

# === App State ===
if "index" not in st.session_state:
    st.session_state.index = 0

def current_row():
    return data.iloc[st.session_state.index]

def _mask_for_instance(df: pd.DataFrame, instance_id: str) -> pd.Series:
    if INSTANCE_COL not in df.columns:
        df[INSTANCE_COL] = ""
    return pd.Series(True, index=df.index) & (df[INSTANCE_COL].astype(str) == _s(instance_id))

def get_saved_entry_for(row):
    instance_id = _s(row.get(INSTANCE_COL, ""))
    mask = _mask_for_instance(saved_df, instance_id)
    match = saved_df[mask]
    if not match.empty:
        return _normalize_entry_types(match.iloc[0].to_dict())

    # default blank entry
    base = {
        INSTANCE_COL: instance_id,
        "coder_id": st.session_state.coder_id,
        "celex_number": _s(row.get(ID_COLUMN, "")),
        "Article Number": "",
        "Core Sentence": "",
        "Definition": yes_no[0],
        "Subject": "",
        "Subject type": "",
        "Delegation Binary": yes_no[0],
        "Delegation Type": delegation_types[0],
        "Derogation Binary": yes_no[0],
        "Derogation Details": "",
        "Dilution Binary": yes_no[0],
        "Dilution Details": "",
        "Instrument": "",
        "Instrument Type": "",
        "Domain": "",
        "Object": "",
        "Object Type": "",
        "validated_at": ""
    }
    for fld in VERIFIED_FIELDS:
        base[f"{fld} (found)"] = _s(row.get(fld, "")).strip()
        base[f"{fld} (correct)"] = "No"
    return _normalize_entry_types(base)

def upsert_entry(entry: dict):
    global saved_df
    entry = _normalize_entry_types(entry)
    entry["coder_id"] = st.session_state.coder_id
    entry["validated_at"] = now_iso()

    instance_id = _s(entry.get(INSTANCE_COL, ""))
    mask = _mask_for_instance(saved_df, instance_id)

    if len(saved_df) and mask.any():
        saved_df.loc[mask, ALL_COLUMNS] = [entry.get(c, "") for c in ALL_COLUMNS]
        action = "updated"
    else:
        saved_df = pd.concat([saved_df, pd.DataFrame([entry])[ALL_COLUMNS]], ignore_index=True)
        action = "created"

    saved_df = _dedupe_saved(saved_df)
    saved_df.to_csv(SAVE_FILE, index=False)
    return action

def ensure_state_for_current():
    row = current_row()
    instance_id = _s(row.get(INSTANCE_COL, ""))
    kp = f"f_{instance_id}_"
    if st.session_state.get(kp + "initialized"):
        return

    entry = get_saved_entry_for(row)
    for field in BASE_COLUMNS:
        if field in ["celex_number", "coder_id"]:
            continue
        if field in VERIFIED_FIELDS:
            continue
        st.session_state[kp + field] = entry[field]

    for field in VERIFIED_FIELDS:
        found_val = _s(row.get(field, "")).strip()
        saved_correct = _s(entry.get(f"{field} (correct)", ""))
        saved_val = _s(entry.get(field, "")).strip()
        if saved_correct in {"Yes", "No"}:
            is_correct = (saved_correct == "Yes")
        else:
            is_correct = bool(found_val) and (saved_val == found_val)
        st.session_state[kp + field + "_correct"] = is_correct
        st.session_state[kp + field] = "" if is_correct else saved_val

    st.session_state[kp + "initialized"] = True

def read_entry_from_state():
    row = current_row()
    instance_id = _s(row.get(INSTANCE_COL, ""))
    kp = f"f_{instance_id}_"

    entry = {
        INSTANCE_COL: instance_id,
        "coder_id": st.session_state.coder_id,
        "celex_number": _s(row.get(ID_COLUMN, "")),
    }
    for field in VERIFIED_FIELDS:
        found_val = _s(row.get(field, "")).strip()
        is_correct = bool(st.session_state.get(kp + field + "_correct", False))
        final_val = found_val if is_correct else _s(st.session_state.get(kp + field, "")).strip()
        entry[field] = final_val
        entry[f"{field} (found)"] = found_val
        entry[f"{field} (correct)"] = "Yes" if is_correct else "No"

    for field in BASE_COLUMNS:
        if field in ["celex_number", "coder_id"] or field in VERIFIED_FIELDS:
            continue
        entry[field] = _s(st.session_state.get(kp + field, ""))

    return _normalize_entry_types(entry)

def core_sentence_variants_for_current(row):
    celex = _s(row.get(ID_COLUMN, "")).strip()
    art = _s(row.get(ARTICLE_COLUMN, "")).strip()
    mask = (data[ID_COLUMN].astype(str).str.strip() == celex)
    if art:
        mask &= (data[ARTICLE_COLUMN].astype(str).str.strip() == art)
    vals = [_s(v).strip() for v in data.loc[mask, CORE_SENTENCE].tolist() if _s(v).strip()]
    seen, uniq = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq

# === Prepare ===
row = current_row()
ensure_state_for_current()
celex = _s(row.get(ID_COLUMN, ""))
kp = f"f_{_s(row.get(INSTANCE_COL, ''))}_"

# === Sidebar (progress + quick navigation) ===
st.sidebar.title("Progress")
st.sidebar.write(f"Document {st.session_state.index + 1} / {len(data)}")
st.sidebar.progress((st.session_state.index + 1) / len(data))
st.sidebar.caption(f"Your file: {os.path.basename(SAVE_FILE)} · Sidebar navigation does not save.")

# Prev / Next (no save)
cprev, cnext = st.sidebar.columns(2)
with cprev:
    if st.button("⏮ Prev", key="sb_prev"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
            safe_rerun()
with cnext:
    if st.button("Next ⏭", key="sb_next"):
        if st.session_state.index < len(data) - 1:
            st.session_state.index += 1
            safe_rerun()

st.sidebar.divider()
# Jump to (1-based)
jump_to = st.sidebar.number_input(
    "Jump to #",
    min_value=1,
    max_value=len(data),
    value=st.session_state.index + 1,
    step=1,
    key="sb_jump_to",
)
if st.sidebar.button("Go", key="sb_go"):
    target = int(jump_to) - 1
    if 0 <= target < len(data):
        st.session_state.index = target
        safe_rerun()

# === Card renderer (used for both Law Text and Core Sentence) ===
def render_card(title_prefix: str, body_text: str, article_suffix: bool = False):
    suffix = ""
    if article_suffix:
        article_num = _s(row.get(ARTICLE_COLUMN, "")).strip()
        suffix = f" (Article {article_num})" if article_num else ""
    body = html_escape(_s(body_text))
    st.markdown(
        f"""<div class="core-box">
               <div class="core-title">{html_escape(title_prefix)}{suffix}</div>
               <div class="core-body">{body}</div>
            </div>""",
        unsafe_allow_html=True
    )

# Helper to sync Verify table back to session_state BEFORE saving
def apply_verify_df_to_state(verify_df: pd.DataFrame):
    if verify_df is None or verify_df.empty:
        return
    instance_id = _s(row.get(INSTANCE_COL, ""))
    base_kp = f"f_{instance_id}_"
    for _, r in verify_df.iterrows():
        fld = _s(r.get("Field", "")).strip()
        if not fld:
            continue
        correct = bool(r.get("Correct?", False))
        st.session_state[base_kp + fld + "_correct"] = correct
        st.session_state[base_kp + fld] = "" if correct else _s(r.get("Correction", "")).strip()

# === Form (tabs first, navigation at the bottom) ===
with st.form("annotation_form", clear_on_submit=False):
    tab_doc, tab_verify, tab_ops = st.tabs(
        ["🧾 Document", "✅ Verify Labels", "⚙️ Delegation / Derogation / Dilution"]
    )

    # --- DOCUMENT TAB ---
    with tab_doc:
        st.subheader(f"CELEX: {celex}")
        st.markdown(
            """<div class="callout info">
            <b>What to do:</b> Compare the <i>Core Sentence</i> with the full <i>Law Text</i>.
            If the core sentence is correct, tick the box. If not, paste a corrected version.
            </div>""",
            unsafe_allow_html=True
        )

        # Same width columns; Core Sentence LEFT, Law Text RIGHT
        left, right = st.columns([1,1], gap="small")
        with left:
            render_card("Core Sentence", _s(row.get(CORE_SENTENCE, "")), article_suffix=True)
            st.checkbox("Core sentence correctly extracted", key=kp + CORE_SENTENCE + "_correct")
            if not st.session_state.get(kp + CORE_SENTENCE + "_correct", False):
                st.text_area("Correction for Core Sentence", key=kp + CORE_SENTENCE, height=120)

        with right:
            render_card("Law Text", _s(row.get(TEXT_COLUMN, "")), article_suffix=False)

        # Multiple core sentence variants indicator (per CELEX + Article)
        variants = core_sentence_variants_for_current(row)
        if len(variants) > 1:
            st.markdown(f'<span class="badge">Multiple core sentences detected: {len(variants)}</span>', unsafe_allow_html=True)
            with st.expander("See all detected core sentences for this article"):
                for i, v in enumerate(variants, 1):
                    st.write(f"{i}. {v if len(v)<=500 else v[:500]+'…'}")

    # --- VERIFY TAB (data editor) ---
    with tab_verify:
        st.markdown(
            """<div class="callout verify">
            <b>What to do:</b> For each label, leave <i>Correct?</i> checked if the detected value is right.
            If not, uncheck it and type a <i>Correction</i>. Use “Hide labels with no detected value” to reduce clutter.
            </div>""",
            unsafe_allow_html=True
        )

        render_card("Core Sentence", _s(row.get(CORE_SENTENCE, "")), article_suffix=True)

        # Build the editable table from current state
        rows = []
        for fld in FIELDS_TO_VERIFY:
            found_val = _s(row.get(fld, "")).strip()
            is_correct = bool(st.session_state.get(kp + fld + "_correct", False))
            correction_val = _s(st.session_state.get(kp + fld, "")).strip()
            rows.append({"Field": fld, "Found": found_val, "Correct?": is_correct, "Correction": correction_val})
        verify_df = pd.DataFrame(rows)

        hide_empty = st.checkbox("Hide labels with no detected value", value=True, key=kp+"hide_empty_verify")
        if hide_empty and not verify_df.empty:
            mask = (verify_df["Found"] != "") | (verify_df["Correction"] != "") | (~verify_df["Correct?"])
            verify_df_display = verify_df[mask].reset_index(drop=True)
        else:
            verify_df_display = verify_df

        verify_df_edited = st.data_editor(
            verify_df_display,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Field": st.column_config.TextColumn("Field", disabled=True, width="medium"),
                "Found": st.column_config.TextColumn("Found (from CSV)", disabled=True),
                "Correct?": st.column_config.CheckboxColumn("Correct?"),
                "Correction": st.column_config.TextColumn("Correction (if incorrect)"),
            },
            key=kp + "verify_editor"
        )

    # --- OPS TAB ---
    with tab_ops:
        st.markdown(
            """<div class="callout ops">
            <b>What to do:</b> Set the <i>Definition</i>, <i>Delegation</i>, <i>Derogation</i>, and <i>Dilution</i> flags.
            Fill the <i>Details</i> fields only when “Yes” or when clarification is needed.
            </div>""",
            unsafe_allow_html=True
        )

        render_card("Core Sentence", _s(row.get(CORE_SENTENCE, "")), article_suffix=True)
        st.caption("Structured inputs.")
        o1, o2 = st.columns(2, gap="large")
        with o1:
            st.selectbox("Definition", yes_no, key=kp + "Definition")
            st.selectbox("Delegation Binary", yes_no, key=kp + "Delegation Binary")
            st.selectbox("Delegation Type", delegation_types, key=kp + "Delegation Type")
        with o2:
            st.selectbox("Derogation Binary", yes_no, key=kp + "Derogation Binary")
            st.text_input("Derogation Details", key=kp + "Derogation Details")
            st.selectbox("Dilution Binary", yes_no, key=kp + "Dilution Binary")
            st.text_input("Dilution Details", key=kp + "Dilution Details")

    # --- Navigation (BOTTOM) ---
    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1], gap="small")
    with c1:  prev_clicked = st.form_submit_button("⏮ Previous", use_container_width=True)
    with c2:  save_clicked = st.form_submit_button("💾 Save", use_container_width=True)
    with c3:  next_clicked = st.form_submit_button("Save & Next ⏭", use_container_width=True)

    # Apply verify table edits to state BEFORE saving
    if save_clicked or next_clicked:
        if 'verify_df_edited' in locals() and verify_df_edited is not None:
            edited_map = {r["Field"]: r for _, r in verify_df_edited.iterrows()}
            merged = []
            for _, r in verify_df.iterrows():
                fld = r["Field"]
                merged.append(edited_map.get(fld, r))
            verify_df_final = pd.DataFrame(merged)
            apply_verify_df_to_state(verify_df_final)

    if save_clicked:
        action = upsert_entry(read_entry_from_state())
        st.success(f"✅ Entry {action}. Saved to {os.path.basename(SAVE_FILE)} (1 row per instance).")

    if next_clicked:
        action = upsert_entry(read_entry_from_state())
        st.success(f"✅ Entry {action}. Moving to next…")
        if st.session_state.index < len(data) - 1:
            st.session_state.index += 1
        safe_rerun()

    if prev_clicked:
        if st.session_state.index > 0:
            st.session_state.index -= 1
        safe_rerun()

# === Saved section (outside form) ===
st.markdown("---")
with st.expander(
    f"📄 Saved codings — {os.path.basename(SAVE_FILE)} "
    f"(linked rows: {len(saved_df)} of {len(data)} instances) — click to expand",
    expanded=False
):
    st.caption("Exactly one row per instance (`__instance_id`) in *your* file. Re-saving the same instance overwrites your previous row.")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        show_current_only = st.checkbox("Show current CELEX only", value=False, key="show_current_only")
    with colB:
        quick_filter = st.text_input("Quick filter (contains…)", value="", placeholder="Search text in any column…")
    with colC:
        if st.button("🗑️ Reset my validation file (start empty)"):
            empty = pd.DataFrame(columns=ALL_COLUMNS)
            empty.to_csv(SAVE_FILE, index=False)
            st.success(f"Cleared {os.path.basename(SAVE_FILE)} — it is now empty.")
            safe_rerun()

    show_df = saved_df if not show_current_only else saved_df[saved_df["celex_number"] == celex]

    if quick_filter.strip():
        q = quick_filter.strip().lower()
        mask = show_df.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False))
        show_df = show_df[mask.any(axis=1)]

    show_df = show_df.sort_values(by=INSTANCE_COL, key=lambda s: pd.to_numeric(s, errors="coerce"))
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=320)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download displayed CSV",
        data=csv_bytes,
        file_name=os.path.basename(SAVE_FILE) if not show_current_only else f"{os.path.splitext(os.path.basename(SAVE_FILE))[0]}_{celex}.csv",
        mime="text/csv"
    )
