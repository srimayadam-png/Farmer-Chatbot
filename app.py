
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

load_dotenv()
st.set_page_config(page_title="Farmer AI Assistant", page_icon="🌱")
st.title("🌱 Farmer AI Assistant")
st.warning("AI advice may contain mistakes. Verify diagnoses and treatments with an agricultural expert.")

@st.cache_resource
def load_resources():
    docs = TextLoader("data/agriculture.txt", encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_documents(chunks, embeddings)
    model = tf.keras.models.load_model("crop_disease_model.keras")
    with open("class_names.json", encoding="utf-8") as f:
        names = json.load(f)
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
    return db, model, names, llm

try:
    vectorstore, model, class_names, llm = load_resources()
except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

def get_guidance(query):
    docs = vectorstore.similarity_search(query, k=3)
    with st.expander("View retrieved agricultural references"):
        for i, doc in enumerate(docs, 1):
            st.write(f"**Reference {i}**")
            st.write(doc.page_content)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_template("""
You are a friendly agricultural assistant helping farmers.
Question: {question}
Agricultural references: {context}
Use simple English, headings and bullet points.
Explain the problem, symptoms, prevention, management and treatment safety.
Distinguish reference-supported facts from general knowledge.
Do not invent pesticide names, doses or chemical recommendations.
Say honestly when specific information is missing.
Do not claim an image prediction is certainly correct.
Finish with this disclaimer:
⚠️ **AI Disclaimer:** This AI-generated advice may contain mistakes and may not correctly identify every plant disease. Verify the information with a qualified agricultural expert before making important crop-treatment decisions.
""")
    response = llm.invoke(prompt.format(question=query, context=context))
    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and "text" in item
        )
    return content

uploaded = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded plant image", width=350)
    if st.button("Detect Disease"):
        try:
            arr = np.array(image.resize((224, 224)), dtype=np.float32)
            arr = preprocess_input(arr)
            predictions = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
            index = int(np.argmax(predictions))
            disease = class_names[index]
            st.subheader("🌿 Disease Prediction")
            st.write(f"**Predicted disease:** {disease}")
            st.metric("Model confidence", f"{predictions[index] * 100:.2f}%")
            st.caption("Confidence is not diagnostic certainty.")
            with st.spinner("Preparing agricultural guidance..."):
                st.markdown(get_guidance(f"Explain symptoms, prevention and safe management of {disease}."))
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.divider()
st.subheader("💬 Ask an Agricultural Question")
question = st.text_input("What would you like to know about your crops?")
if st.button("Get Answer"):
    if question.strip():
        try:
            with st.spinner("Finding agricultural information..."):
                answer = get_guidance(question.strip())
            st.markdown(answer)
        except Exception as e:
            st.error(f"Answer error: {e}")
    else:
        st.warning("Please enter a question.")
