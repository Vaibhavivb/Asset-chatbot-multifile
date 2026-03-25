import streamlit as st
import pdfplumber
import cohere
import json
import pandas as pd
import re
import hashlib
from rapidfuzz import fuzz

# ================= CONFIG =================
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

# ================= CLEAN JSON =================
def extract_data(text):
    prompt = f"""
    Extract asset details in STRICT JSON only.

    {{
      "asset_name": "...",
      "location": "...",
      "current_status": "...",
      "last_maintenance_date": "...",
      "risk_level": "..."
    }}

    If missing → "NOT FOUND"

    Text:
    {text[:3000]}
    """

    response = call_llm(prompt)
    response = response.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())

        # Normalize
        for k in data:
            if isinstance(data[k], str):
                data[k] = data[k].lower().strip()

        # Generate ID
        data["asset_id"] = generate_asset_id(data)

        return data

    except:
        return None

# ================= SEARCH =================
def search_metadata(df, query):
    query = query.lower()

    results = []

    for _, row in df.iterrows():
        score = 0

        # keyword match
        for col in df.columns:
            val = str(row[col])
            if query in val:
                score += 2

        # fuzzy match
        score += fuzz.partial_ratio(query, row.get("asset_name","")) / 50

        if score > 1:
            results.append((score, row.to_dict()))

    results.sort(reverse=True, key=lambda x: x[0])

    return [r[1] for r in results[:5]]

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
    for file in uploaded_files:
        text = extract_pdf(file)
        data = extract_data(text)

        if data:
            st.session_state.metadata_store.append(data)

    st.success(f"✅ Processed {len(uploaded_files)} files")

# ================= DATAFRAME =================
df = pd.DataFrame(st.session_state.metadata_store)

# ================= CHAT =================
user_input = st.chat_input("Ask about assets...")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    # ---------- RULE BASED SEARCH ----------
    if len(df) > 0:
        results = search_metadata(df, user_input)

        if results:
            bot_reply = results

        else:
            # fallback to LLM
            context = df.to_string()

            prompt = f"""
            Answer based on this asset dataset:

            {context[:4000]}

            Question: {user_input}
            """

            bot_reply = call_llm(prompt)

    else:
        bot_reply = "⚠️ No data available. Upload PDFs first."

    with st.chat_message("assistant"):
        st.write(bot_reply)
