import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Multilingual NER")

st.title("🧠 Multilingual Named Entity Recognition (NER)")
st.markdown("Powered by `Davlan/bert-base-multilingual-cased-ner-hrl` 🤗")

@st.cache_resource
def load_pipeline():
    return pipeline(
        "ner",
        model="Davlan/bert-base-multilingual-cased-ner-hrl",
        tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
        aggregation_strategy="simple"
    )

ner_pipeline = load_pipeline()

text = st.text_area("Enter your text:", height=150)

if st.button("Extract Entities"):
    if text:
        with st.spinner("Extracting..."):
            results = ner_pipeline(text)
        for ent in results:
            st.markdown(f"- **Entity**: `{ent['word']}` — *{ent['entity_group']}* ({ent['score']:.2f})")
    else:
        st.warning("Please enter some text.")

