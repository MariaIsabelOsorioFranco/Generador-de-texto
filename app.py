import streamlit as st
import numpy as np
import json
import platform

# Intentar cargar tf_keras para compatibilidad con modelos antiguos
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

st.set_page_config(
    page_title="Generador LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #764ba2; }
    .generated-text { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; font-family: Georgia, serif; line-height: 1.8; border-left: 5px solid #764ba2; }
    .stApp { background-color: #fcfcfc; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        model = keras.models.load_model(model_path, compile=False)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)

def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(model, seed, char_to_idx, idx_to_char, seq_len, n_chars, temp):
    generated = seed
    for _ in range(n_chars):
        x = np.array([char_to_idx.get(c, 0) for c in generated[-seq_len:]])
        x = x.reshape(1, len(x))
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temp)]
        generated += next_char
    return generated[len(seed):]

# UI
st.markdown('<h1 class="main-title">🧠 Generador de Texto LSTM</h1>', unsafe_allow_html=True)
st.write(f"Versión Python: {platform.python_version()}")

with st.sidebar:
    st.header("Configuración")
    model_file = st.file_uploader("Modelo (.keras o .h5)", type=["keras", "h5"])
    meta_file = st.file_uploader("Metadatos (.json)", type=["json"])
    temp = st.slider("Temperatura", 0.1, 2.0, 0.8)

if model_file and meta_file:
    with open("temp_model.keras", "wb") as f: f.write(model_file.getvalue())
    with open("temp_meta.json", "wb") as f: f.write(meta_file.getvalue())
    
    model, meta, err = load_model_and_metadata("temp_model.keras", "temp_meta.json")
    
    if model:
        seed = st.text_area("Texto semilla", "en un lugar de la mancha")
        if st.button("Generar"):
            res = generate_text(model, seed.lower(), meta["char_to_idx"], 
                               meta["idx_to_char"], meta["seq_length"], 200, temp)
            st.markdown(f'<div class="generated-text">{seed + res}</div>', unsafe_allow_html=True)
    else:
        st.error(f"Error: {err}")
