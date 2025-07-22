import streamlit as st
import openai
import pickle
from openai import OpenAI
import numpy as np

# Load OpenAI key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load saved model
@st.cache_resource
def load_model():
    with open("rag_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]

index, chunks = load_model()

st.title("📄 RAG Q&A from Preprocessed PDF")

question = st.text_input("Ask a question:")
if question:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    q_emb = embed_model.encode([question])
    _, I = index.search(np.array(q_emb), 3)
    context = "\n".join([chunks[i] for i in I[0]])

    with st.spinner("Generating answer..."):
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        st.write("### 🧠 Answer:")
        st.write(answer)