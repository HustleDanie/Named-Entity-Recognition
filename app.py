import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ✅ Set Streamlit page config
st.set_page_config(page_title="Multilingual NER App", layout="centered")

# ✅ Title
st.title("Hustle's Project: Multilingual Named Entity Recognition")


# ✅ Cache model loading
@st.cache_resource
def load_ner_pipeline():
    model_name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner_pipeline = load_ner_pipeline()

# ✅ User input
text = st.text_area("Enter text in any supported language:", height=200, placeholder="Type something like: Angela Merkel war die Bundeskanzlerin von Deutschland.")

# ✅ Run NER
if st.button("🔍 Recognize Entities") and text:
    with st.spinner("Analyzing..."):
        try:
            entities = ner_pipeline(text)
            if not entities:
                st.info("No named entities found.")
            else:
                st.subheader("📌 Detected Entities")
                for ent in entities:
                    st.markdown(f"• **{ent['word']}** — *{ent['entity_group']}* (Score: {ent['score']:.2f})")
        except Exception as e:
            st.error(f"Error: {e}")
