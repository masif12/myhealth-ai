import streamlit as st
from app.graph.medical_graph import app_graph
from app.chains.memory_rag import create_rag_chain
import base64
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="🧠 MyHealthAI", layout="wide", page_icon="🧠")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    body {
        background-color: #F7F9FB;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #004C99;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<h1 style='text-align: center; color:#333;'>🧠 MyHealthAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:#555;'>Your Multimodal AI Medical Assistant</h4>", unsafe_allow_html=True)
st.markdown("#### ", unsafe_allow_html=True)

st.markdown("---")

# ---------- Input and Output Columns ----------
col1, col2 = st.columns(2, gap="large")

# ---------- Input Section ----------
with col1:
    st.subheader("📝 Patient Input")
    with st.container():
        symptoms = st.text_area("🤒 Describe your symptoms", height=150, placeholder="e.g., chest pain, high fever, nausea")

        # Temporarily disabling image upload due to API quota limits
        st.info(
            "⚠️ Image upload feature is temporarily disabled due to API usage limits. "
            "Please provide symptoms only. "
            "We apologize for the inconvenience and will re-enable this feature soon."
        )
        # Commented out image upload widget:
        # uploaded_img = st.file_uploader("📸 Upload Medical Image (optional)", type=["jpg", "jpeg", "png"])
        uploaded_img = None

        st.markdown(" ")
        run_button = st.button("🔍 Diagnose Symptoms")

# ---------- Output Section ----------
with col2:
    st.subheader("🧠 Diagnosis Output")
    with st.container():
        if run_button:
            # Require symptoms since image upload is disabled
            if not symptoms:
                st.warning("⚠️ Please enter your symptoms to proceed.")
            else:
                img_base64 = ""
                # Image upload disabled, so no base64 encoding
                # if uploaded_img:
                #     img_base64 = base64.b64encode(uploaded_img.read()).decode("utf-8")

                state = {"symptoms": symptoms, "image_base64": img_base64}

                with st.spinner("🔬 Running AI agents for diagnosis..."):
                    try:
                        result = app_graph.invoke(state)
                        st.success("✅ Diagnosis complete!")

                        diagnosis = result.get("diagnosis", "❓ No diagnosis found.")
                        treatment = result.get("treatment", "❓ No treatment suggestion.")

                        st.markdown(f"**🩺 Diagnosis:** `{diagnosis}`")
                        st.markdown(f"**💊 Treatment Suggestion:** `{treatment}`")
                    except Exception as e:
                        st.error(f"❌ Error occurred: {e}")
        else:
            st.info("🔄 Awaiting input to begin diagnosis.")

# ---------- RAG Section ----------
st.markdown("---")
st.subheader("📄 Ask AI About Medical Documents")

rag_col1, rag_col2 = st.columns(2, gap="large")

with rag_col1:
    with st.container():
        doc_file = st.file_uploader("📥 Upload a document (PDF, TXT, CSV)", type=["pdf", "txt", "csv"])
        if doc_file:
            os.makedirs("temp_docs", exist_ok=True)
            path = os.path.join("temp_docs", doc_file.name)
            with open(path, "wb") as f:
                f.write(doc_file.getbuffer())
            st.success("✅ Document uploaded successfully!")
            st.session_state.rag_chain = create_rag_chain(path)

with rag_col2:
    with st.container():
        if "rag_chain" in st.session_state:
            question = st.text_input("💬 Ask about your document")
            if question:
                with st.spinner("📖 Searching document..."):
                    rag_func, _ = st.session_state.rag_chain
                    answer = rag_func(question)
                st.markdown("### 🤖 Answer:")
                st.success(answer)
        else:
            st.info("⬅️ Upload a document first.")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        🚀 Built with <strong>LangGraph</strong>, <strong>LangChain</strong>, <strong>Streamlit</strong>, and <strong>GPT-4o</strong><br>
        👨‍⚕️ Created by <strong>Muhammad Asif</strong> | © 2025 <strong>MyHealthAI</strong>
    </div>
    """,
    unsafe_allow_html=True
)
