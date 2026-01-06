import json
import os
import requests
import time
import faiss
import streamlit as st
from typing import List
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ==================== STREAMLIT CONFIG ====================
st.set_page_config(
    page_title="KIBA",
    page_icon="🧠",
    layout="centered"
)

st.subheader("by Evans")

st.markdown("""
<style>
/* BODY */
body { background-color:#0b0f19; color:#e6e6e6; font-family: 'Segoe UI', sans-serif; }

/* TEXTAREA */
.stTextArea textarea {
    background: rgba(255,255,255,0.05);
    color:#e6e6e6;
    border:1px solid rgba(255,255,255,0.2);
    border-radius:8px;
    padding:6px;
}

/* BUTTON */
.stButton button {
    background: rgba(255,255,255,0.05);
    color:#e6e6e6;
    border:1px solid rgba(255,255,255,0.2);
    border-radius:8px;
    font-weight:bold;
}
.stButton button:hover {
    background: rgba(255,255,255,0.1);
}

/* CHAT BUBBLES */
.user {
    background: rgba(255,255,255,0.05);
    padding:10px; border-radius:12px; margin-bottom:8px;
    border-left: 4px solid rgba(255,255,255,0.2);
}
.ai {
    background: rgba(255,255,255,0.08);
    padding:10px; border-radius:12px; margin-bottom:12px;
    border-left: 4px solid rgba(255,255,255,0.3);
}

/* SCROLLBAR */
::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ==================== SECURITY & RATE LIMITING ====================
rate_limits = defaultdict(list)
BLOCKED = [
    'harmful','dangerous','illegal','hate speech','violence','terrorism',
    'self-harm','suicide','exploit','malware','phishing','scam','fraud',
    'porn','nsfw','racist','sexist','discriminatory','harassment','bullying'
]

HF_API_KEY = os.getenv("HF_API_KEY")

valid_token = {}
token_counter = 0

def check_rate_limit(user_id, limit=2, per=60):
    now = time.time()
    rate_limits[user_id] = [t for t in rate_limits[user_id] if now - t < per]
    if len(rate_limits[user_id]) >= limit:
        return False
    rate_limits[user_id].append(now)
    return True

def check_safety(text):
    return not any(b in text.lower() for b in BLOCKED)

def create_token(user_id):
    global token_counter
    token_counter += 1
    token = f"token_{token_counter}_{user_id}"
    valid_token[token] = {
        "user_id": user_id,
        "expires": time.time() + 1200
    }
    return token

def verify_token(token):
    data = valid_token.get(token)
    if not data or time.time() > data["expires"]:
        valid_token.pop(token, None)
        return None
    return data

def secure_request(user_id, text):
    if not check_rate_limit(user_id):
        return {"error": "Rate limit exceeded"}
    if not check_safety(text):
        return {"error": "Content blocked"}
    return {"ok": True}

# ==================== LOAD CHUNKS ====================
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ==================== RAG SETUP ====================
embedding_model = None
faiss_index = None
document_chunks = []

def init_rag(model_path="all-MiniLM-L6-v2"):
    global embedding_model
    embedding_model = SentenceTransformer(model_path)

def build_knowledge_base(docs):
    global faiss_index, document_chunks
    document_chunks = docs
    emb = embedding_model.encode(docs)
    import numpy as np
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatL2(emb.shape[1])
    faiss_index.add(emb.astype("float32"))

def retrieve_documents(q, k=3):
    if faiss_index is None:
        return []
    emb = embedding_model.encode([q])
    import numpy as np
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _, idx = faiss_index.search(emb.astype("float32"), k)
    return [document_chunks[i] for i in idx[0] if i < len(document_chunks)]

def rag_generate(llm, query, history):
    docs = retrieve_documents(query)
    context = "\n".join(docs) if docs else "No relevant context found."

    conv = ""
    for m in history[-2:]:
        conv += f"{m['role'].capitalize()}: {m['content']}\n"

    prompt = f"""
Context:
{context}

Instruction:
If no relevant context is found, answer independently.
Do NOT repeat the user's question.
Do NOT mention missing context.

Previous conversation:
{conv}

User: {query}
Rewrite into a complete, professional 90–100 word answer. End with a full stop.
"""
    return llm(prompt)["generated_text"]

# ==================== CHAT SYSTEM ====================
def init_chat(llm):
    return {"llm": llm, "history": []}

def chat(sys, msg):
    sys["history"].append({"role":"user","content":msg})
    reply = rag_generate(sys["llm"], msg, sys["history"][:-1])
    sys["history"].append({"role":"assistant","content":reply})
    return reply

# ==================== LLM CLIENT ====================
def query_llama(prompt):
    r = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization":"Bearer {HF_API_KEY}"},
        json={
            "model":"Qwen/Qwen2.5-Coder-7B-Instruct",
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.1
        }
    ).json()
    try:
        return {"generated_text": r["choices"][0]["message"]["content"]}
    except:
        return {"generated_text": "Model error."}

# Your existing functions (assumed to be defined elsewhere)
# create_token, init_chat, query_llama, init_rag, build_knowledge_base, chat

# ==================== Initialize ====================
if embedding_model is None:
    init_rag()
    build_knowledge_base(chunks)

if "token" not in st.session_state:
    st.session_state.token = create_token("kiba123")

if "chat" not in st.session_state:
    st.session_state.chat = init_chat(query_llama)

if "log" not in st.session_state:
    st.session_state.log = []

# ==================== UI ====================
st.markdown("## 🧠 GlobalTech Inquiry Bot")

# Display chat history
for role, message in st.session_state.log:
    css_class = "user" if role == "user" else "ai"
    st.markdown(f"<div class='{css_class}'>{message}</div>", unsafe_allow_html=True)

# ==================== Chat Input ====================
# Using a form for better UX and Enter key support
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Message", height=80, key="input_text")
    submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input.strip():
        # Add user message to log
        st.session_state.log.append(("user", user_input))
        
        # Get AI response
        try:
            raw_response = chat(st.session_state.chat, user_input)
        except Exception as e:
            raw_response = f"Error: {str(e)}"
        
        # Stream the response
        response_placeholder = st.empty()
        streamed_response = ""
        
        # Split response into words for streaming effect
        words = raw_response.split()
        for i, word in enumerate(words):
            streamed_response += word + " "
            response_placeholder.markdown(
                f"<div class='ai'>{streamed_response}</div>", 
                unsafe_allow_html=True
            )
            time.sleep(0.02)  # Adjust speed as needed
        
        # Add final response to log
        st.session_state.log.append(("ai", streamed_response))
        
        # Rerun to show updated chat and clear form
        st.rerun()
