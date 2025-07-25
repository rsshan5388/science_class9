import streamlit as st
import pickle
import faiss
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load OpenAI API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_model():
    with open("class9science.pkl", "rb") as f:
        data = pickle.load(f)
    buf = io.BytesIO(data["index_bytes"])
    index = faiss.read_index(faiss.PyCallbackIOReader(buf.read))
    return index, data["chunks"]

# Load RAG index and data
index, chunks = load_model()

# Load embedder
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📚 Class 9 Science Q&A RAG Module")
question = st.text_input("Ask a question about science:")

if question:
    st.info("Searching the document...")

    q_embedding = embed_model.encode([question])
    _, I = index.search(np.array(q_embedding), 3)
    context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"Use the context to answer:\n\n{context}\n\nQuestion: {question}"

    with st.spinner("Generating answer..."):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        st.success("Answer:")
        st.write(answer)
