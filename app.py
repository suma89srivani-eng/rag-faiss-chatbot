import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# Load embedding model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# -------------------------------
# Convert text to embeddings
# -------------------------------
def generate_embeddings(texts, model):
    return model.encode(texts)


# -------------------------------
# Load and prepare documents
# -------------------------------
@st.cache_data
def load_documents(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    # Combine Q + A
    documents = [f"{lines[i]} {lines[i+1]}" for i in range(0, len(lines), 2)]
    return documents


# -------------------------------
# Create FAISS index (FIXED ERROR HERE)
# -------------------------------
@st.cache_resource
def create_faiss_index(documents, _model):   # 👈 underscore fix
    embeddings = generate_embeddings(documents, _model)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings


# -------------------------------
# Retrieve answers
# -------------------------------
def retrieve(query, model, index, documents, top_k=2):
    query_embedding = generate_embeddings([query], model)

    D, I = index.search(np.array(query_embedding), top_k)

    results = [documents[i] for i in I[0]]
    return results


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("📚 RAG Chatbot using FAISS")

    uploaded_file = st.file_uploader("Upload your Employee.txt file", type="txt")

    if uploaded_file:
        filepath = "temp.txt"

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully ✅")

        # Load everything
        model = load_model()
        documents = load_documents(filepath)
        index, _ = create_faiss_index(documents, model)

        st.write(f"Loaded {len(documents)} documents")

        # User query
        query = st.text_input("Ask a question:")

        if query:
            results = retrieve(query, model, index, documents)

            st.subheader("📌 Answer:")
            for res in results:
                st.write(res)


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    main()
