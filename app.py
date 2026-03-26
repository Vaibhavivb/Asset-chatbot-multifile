import streamlit as st
import pdfplumber
import cohere
import json
import pandas as pd
import re
import hashlib
from rapidfuzz import fuzz

# ================= CONFIG =================
if "COHERE_API_KEY" not in st.secrets:
    st.error("Missing COHERE_API_KEY in secrets.toml")
    st.stop()

co = cohere.Client(st.secrets["COHERE_API_KEY"])

# ================= LLM =================
def call_llm(prompt):
    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )
    return response.text

# ================= PDF =================
def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ================= SYNTHETIC ID =================
def generate_asset_id(data):
    key = f"{data.get('asset_name','')}_{data.get('location','')}"
    return hashlib.md5(key.encode()).hexdigest()

# ================= MULTI-ASSET EXTRACTION =================
def extract_data(text):
    prompt = f"""
Extract ALL assets from the document.

IMPORTANT:
- There may be multiple assets
- Each asset must be separate
- Do NOT merge assets
- If only one asset exists → return list with 1 item

Return STRICT JSON ARRAY:

[
  {{
    "asset_name": "...",
    "location": "...",
    "current_status": "...",
    "last_maintenance_date": "...",
    "risk_level": "..."
  }}
]

If missing → "NOT FOUND"

Text:
{text[:4000]}
"""

    response = call_llm(prompt)
    response = response.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        return []

    try:
        data_list = json.loads(match.group())

        for data in data_list:
            for k in data:
                if isinstance(data[k], str):
                    data[k] = data[k].lower().strip()

            data["asset_id"] = generate_asset_id(data)

        return data_list

    except:
        return []

# ================= SEARCH =================
def search_metadata(df, query):
    query = query.lower()
    results = []

    for _, row in df.iterrows():
        score = 0

        for col in df.columns:
            val = str(row[col])
            if query in val:
                score += 2

        score += fuzz.partial_ratio(query, row.get("asset_name", "")) / 50

        if score > 1:
            results.append((score, row.to_dict()))

    results.sort(reverse=True, key=lambda x: x[0])
    return [r[1] for r in results[:5]]

# ================= DEDUP =================
def add_unique_asset(data):
    existing_ids = [d["asset_id"] for d in st.session_state.metadata_store]

    if data["asset_id"] not in existing_ids:
        st.session_state.metadata_store.append(data)

# ================= INIT =================
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = []

# ================= UI =================
st.title("🤖 Asset Intelligence Chatbot")

uploaded_files = st.file_uploader(
    "Upload Asset PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ================= PROCESS FILES =================
if uploaded_files:
    st.session_state.metadata_store = []

    for file in uploaded_files:
        try:
            text = extract_pdf(file)
            data_list = extract_data(text)

            for data in data_list:
                if data:
                    add_unique_asset(data)

        except:
            st.warning(f"⚠️ Failed to process {file.name}")

    st.success(f"✅ Processed {len(uploaded_files)} files")

# ================= DATAFRAME =================
df = pd.DataFrame(st.session_state.metadata_store)

# ================= SHOW TABLE =================
if not df.empty:
    st.subheader("📊 Asset Comparison Table")
    st.dataframe(df)

    # 🔥 Filters
    col1, col2 = st.columns(2)

    with col1:
        risk_filter = st.selectbox(
            "Filter by Risk",
            ["All"] + list(df["risk_level"].dropna().unique())
        )

    with col2:
        location_filter = st.selectbox(
            "Filter by Location",
            ["All"] + list(df["location"].dropna().unique())
        )

    filtered_df = df.copy()

    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

    if location_filter != "All":
        filtered_df = filtered_df[filtered_df["location"] == location_filter]

    st.write("### 🔍 Filtered Results")
    st.dataframe(filtered_df)

    # 🔥 Highlight high risk
    st.write("### ⚠️ High Risk Assets")
    high_risk = df[df["risk_level"].str.contains("high", na=False)]

    if not high_risk.empty:
        st.dataframe(high_risk)
    else:
        st.write("No high risk assets found")

# ================= CHAT =================
user_input = st.chat_input("Ask about assets...")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    if len(df) > 0:

        results = search_metadata(df, user_input)

        if results:
            bot_reply = results

        else:
            context = df.to_string() if not df.empty else "No data"

            prompt = f"""
You are analyzing structured asset data.

IMPORTANT RULES:
- Do NOT assume duplicates
- Do NOT infer multiple assets unless clearly different
- Only count distinct assets
- Answer strictly based on given data

Dataset:
{context[:4000]}

Question: {user_input}
"""

            bot_reply = call_llm(prompt)

    else:
        bot_reply = "⚠️ No data available. Upload PDFs first."

    with st.chat_message("assistant"):
        st.write(bot_reply)
