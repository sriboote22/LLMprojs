import streamlit as st
from transformers import pipeline

# Load Hugging Face Language Detection Model
@st.cache_resource  # Cache the model to avoid reloading
def load_language_identifier():
    return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# Load Hugging Face Translation Pipelines
@st.cache_resource  # Cache the models to avoid reloading
def load_translation_pipeline(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    return pipeline("translation", model=model_name)

# App Title
st.title("Language Detection and Translation App 🌍")

# Input Text
input_text = st.text_area("Enter text to identify and translate:", placeholder="Type something here...")

# Target Language Selection
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
    "Hindi": "hi",
    "Arabic": "ar",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
}
target_language = st.selectbox("Select target language for translation:", list(languages.keys()))

# Process Input
if st.button("Identify and Translate"):
    if input_text.strip():
        try:
            # Detect Language
            language_identifier = load_language_identifier()
            detection = language_identifier(input_text)
            detected_language_code = detection[0]["label"]  # Language code
            confidence = detection[0]["score"]  # Confidence score

            # Map detected language to name
            detected_language_name = next(
                (name for name, code in languages.items() if code == detected_language_code), 
                "Unknown Language"
            )

            # Display Detected Language
            st.write(f"**Detected Language:** {detected_language_name} ({confidence*100:.2f}% confidence)")

            # Translation
            target_lang_code = languages[target_language]
            try:
                # Load the appropriate translation model
                translation_pipeline = load_translation_pipeline(detected_language_code, target_lang_code)
                translation = translation_pipeline(input_text)
                translated_text = translation[0]["translation_text"]
                st.write(f"**Translated Text ({target_language}):** {translated_text}")
            except Exception as e:
                st.error(f"Translation model for: {detected_language_code} to {target_lang_code} not available.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter text to analyze.")