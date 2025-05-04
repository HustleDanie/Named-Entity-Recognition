import streamlit as st
from transformers import pipeline

# Set the page config
st.set_page_config(page_title="Multilingual NER App", layout="centered")

# Title
st.title("🌍 Multilingual Named Entity Recognition (NER)")
st.markdown("This app uses [Davlan's multilingual BERT NER model](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) to extract `PER`, `ORG`, and `LOC` entities from input text.")

# Load the pipeline only once
@st.cache_resource
def load_pipeline():
    return pipeline(
        "ner",
        model="Davlan/bert-base-multilingual-cased-ner-hrl",
        tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
        aggregation_strategy="simple"
    )

ner_pipeline = load_pipeline()

# User input
user_text = st.text_area("✏️ Enter your multilingual text below:", height=150)

# Trigger prediction
if st.button("Extract Entities"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = ner_pipeline(user_text)
        if results:
            st.success("Named Entities Extracted:")
            for ent in results:
                st.markdown(f"""
                    - **Entity**: `{ent['word']}`  
                    - **Type**: `{ent['entity_group']}`  
                    - **Score**: `{ent['score']:.2f}`  
                    - **Start → End**: `{ent['start']}` → `{ent['end']}`
                    ---
                """)
        else:
            st.info("No entities were found in the text.")
