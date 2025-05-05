import streamlit as st
from transformers import pipeline

# Set Streamlit app layout
st.set_page_config(page_title="Lightweight Multilingual NER", layout="centered")

# App title and description
st.title("⚡ Lightweight Multilingual NER")
st.markdown(
    "Uses `Davlan/distilbert-base-multilingual-cased-ner-hrl` to extract **PER**, **ORG**, and **LOC** entities "
    "from text in multiple languages (English, French, Arabic, Chinese, etc.)."
)

# Load the NER model (lightweight version)
@st.cache_resource
def load_ner_pipeline():
    return pipeline(
        "ner",
        model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        tokenizer="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        aggregation_strategy="simple"
    )

ner_pipeline = load_ner_pipeline()

# Text input
user_input = st.text_area("✍️ Enter multilingual text here:", height=150)

# Run entity recognition
if st.button("🔍 Extract Entities"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            entities = ner_pipeline(user_input)

        if entities:
            st.success("Entities Detected:")
            for ent in entities:
                st.markdown(
                    f"• **{ent['entity_group']}** → `{ent['word']}` "
                    f"(Score: `{ent['score']:.2f}`, Pos: {ent['start']}–{ent['end']})"
                )
        else:
            st.info("No entities found.")
