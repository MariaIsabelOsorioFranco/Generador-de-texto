import streamlit as st
import numpy as np
import json
import os

# Configuración de página
st.set_page_config(
    page_title="Generador LSTM Morado",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estética Morada
st.markdown("""
<style>
    .stApp { background-color: #f3e5f5; }
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #7b1fa2 0%, #4a148c 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #4a148c; font-size: 1rem; }
    .generated-text {
        background: #ffffff;
        border-radius: 12px; padding: 1.5rem;
        font-family: Georgia, serif; font-size: 1.05rem;
        line-height: 1.8; color: #311b92;
        border-left: 5px solid #7b1fa2; min-height: 120px;
    }
    .info-box {
        background: #e1bee7; border-radius: 8px; padding: 1rem;
        border: 1px solid #ce93d8; font-size: 0.9rem; color: #4a148c;
    }
    section[data-testid="stSidebar"] { background-color: #f3e5f5; border-right: 1px solid #d1c4e9; }
</style>
""", unsafe_allow_html=True)

# ── Funciones ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)

def is_embedding_model(model):
    first = model.layers[0]
    return hasattr(first, "input_dim") or first.__class__.__name__ == "Embedding"

def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))

def prepare_input(window, char_to_idx, vocab_size, use_embedding):
    indices = [char_to_idx.get(c, 0) for c in window]
    if use_embedding:
        return np.array([indices], dtype=np.int32)
    else:
        x = np.array(indices, dtype=np.float32) / float(vocab_size)
        return x.reshape(1, len(window), 1)

def generate_full_text(model, seed_text, char_to_idx, idx_to_char, seq_length, vocab_size, n_chars=200, temperature=0.8):
    use_emb = is_embedding_model(model)
    seed_text = seed_text.lower()
    if len(seed_text) < seq_length: seed_text = seed_text.rjust(seq_length)
    seed_text = seed_text[-seq_length:]
    seed_text = "".join(c if c in char_to_idx else " " for c in seed_text)
    generated = ""
    window = list(seed_text)
    for _ in range(n_chars):
        x = prepare_input(window[-seq_length:], char_to_idx, vocab_size, use_emb)
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temperature)]
        generated += next_char
        window.append(next_char)
    return generated

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    model_file = st.file_uploader("Modelo (.keras o .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("Metadatos (.json)", type=["json"])
    temperature = st.slider("Temperatura", 0.1, 2.0, 0.8, 0.05)
    n_chars = st.slider("Longitud", 50, 500, 200, 50)
    st.markdown("---")
    st.markdown("""<div class="info-box">Generador LSTM con estética morada.</div>""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🔮 Generador de Texto LSTM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Curso Agentes de IA e Interfaces Multimodales</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Generar", "Comparar", "Teoría"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        seed_input = st.text_area("Texto semilla", "en un lugar de la mancha", height=100)
        gen_btn = st.button("Generar Texto", type="primary", use_container_width=True)
    with col2:
        output = st.empty()
        output.markdown('<div class="generated-text"><em>El texto aparecerá aquí...</em></div>', unsafe_allow_html=True)

    if gen_btn and model_file and metadata_file:
        with open("/tmp/model.tmp", "wb") as f: f.write(model_file.getvalue())
        with open("/tmp/meta.json", "wb") as f: f.write(metadata_file.getvalue())
        model, meta, err = load_model_and_metadata("/tmp/model.tmp", "/tmp/meta.json")
        if model:
            texto = generate_full_text(model, seed_input, meta["char_to_idx"], meta["idx_to_char"], 
                                      meta["seq_length"], meta["vocab_size"], n_chars, temperature)
            output.markdown(f'<div class="generated-text">{texto}</div>', unsafe_allow_html=True)
        else: st.error(err)

with tab2:
    st.markdown("### Comparación de Temperaturas")
    if st.button("Ejecutar Comparación"):
        st.write("Configuración cargada. (Simulación de comparativa activa)")

with tab3:
    st.markdown("### Teoría")
    st.info("El estado oculto actúa como memoria en las RNNs.")

st.markdown("---")
