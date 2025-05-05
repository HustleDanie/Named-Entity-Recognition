import streamlit as st
from transformers import pipeline

# Set Streamlit page configuration
st.set_page_config(page_title="Multilingual NER", layout="centered")

# Title and description
st.title("🌍 Multilingual Named Entity Recognition (NER)")
st.markdown(
    "This app uses [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) "
    "to recognize `PER`, `ORG`, and `LOC` entities across multiple languages."
)

# Load the NER pipeline (cached to avoid reloading on every run)
@st.cache_resource
def load_pipeline():
    return pipeline(
        "ner",
        model="Davlan/bert-base-multilingual-cased-ner-hrl",
        tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
        aggregation_strategy="simple"
    )

ner = load_pipeline()

# User text input
user_input = st.text_area("✏️ Enter text in any supported language (e.g., English, French, Arabic, Chinese):", height=150)

# Process the input
if st.button("🔍 Extract Entities"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            entities = ner(user_input)
        
        if entities:
            st.success("Entities detected:")
            for ent in entities:
                st.markdown(
                    f"• **{ent['entity_group']}** → `{ent['word']}` "
                    f"(Score: `{ent['score']:.2f}`, Pos: {ent['start']}–{ent['end']})"
                )
        else:
            st.info("No entities found.")
