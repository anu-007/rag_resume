import streamlit as st
from src.pipeline import build_pipeline, generate_artifacts
from src.loader import load_image
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Resume", layout="wide")

def settings_page():
    st.header("Settings")

    st.subheader("Models (Entity extraction and summary)")
    llm_model = st.selectbox("LLM Model", ["gpt-3.5-turbo", "gemma3:4b"], key="llm_model_select")

    st.subheader("Retrieval")
    retriever_type = st.selectbox("Retriever Type", ["Ensemble", "BM25", "Chroma"], key="retriever_type_select")
    k = st.slider("Retriever K", min_value=1, max_value=8, value=st.session_state.settings.get('k', 4), key="k_slider")

    st.subheader("Reranker")
    use_reranker = st.checkbox("Use Reranker", value=st.session_state.settings.get('use_reranker', True), key="reranker_checkbox")
    top_n = st.slider("Reranker Top N", min_value=1, max_value=8, value=st.session_state.settings.get('top_n', 4), key="top_n_slider")

    # Save settings to session state
    st.session_state.settings = {
        "llm_model": llm_model,
        "retriever_type": retriever_type,
        "k": k,
        "use_reranker": use_reranker,
        "top_n": top_n,
    }

def chat_page():
    st.header("RAG Resume Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question"):
        if "jd_doc" not in st.session_state:
            st.warning("Please process a job description first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Generating response..."):
                try:
                    settings = st.session_state.settings
                    p = build_pipeline(
                        st.session_state.jd_doc,
                        retriever_type=settings["retriever_type"],
                        k=settings["k"],
                        use_reranker=settings["use_reranker"],
                        top_n=settings["top_n"],
                    )
                    out = generate_artifacts(p, settings["llm_model"], prompt)
                    if prompt == 'resume':
                        message_placeholder.markdown(out)
                    else:
                        message_placeholder.json(out)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    out = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": out})

    with st.sidebar:
        st.subheader("Job Description Input")
        input_method = st.radio("Input Method", ["Image", "Text"], key="input_method_radio")
        settings = st.session_state.settings

        if input_method == "Image":
            jd_image = st.file_uploader("Upload Image File", type=["png", "jpg", "jpeg"], key="image_uploader")
            if jd_image:
                with st.spinner("Processing..."):
                    try:
                        image_bytes = jd_image.read()
                        jd_doc = load_image(image_bytes, jd_image.name)
                        st.session_state.jd_doc = jd_doc
                        st.success("Job description processed!")
                    except RuntimeError as e:
                        st.error(e)
                    except Exception as e:
                        st.error(f"Error processing image file: {e}")
        
        elif input_method == "Text":
            jd_text = st.text_area("Paste Job Description here", height=200, key="jd_text_area")
            if st.button("Process Text", key="process_text_button"):
                if jd_text:
                    with st.spinner("Processing..."):
                        try:
                            jd_doc = Document(page_content=jd_text, metadata={"source": "Text Input"})
                            st.session_state.jd_doc = jd_doc
                            st.success("Job description processed!")
                        except Exception as e:
                            st.error(f"Error processing text: {e}")
                else:
                    st.warning("Please paste the job description text.")

def main():
    PAGES = {
        "Chat": chat_page,
        "Settings": settings_page,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="nav_radio")

    if 'settings' not in st.session_state:
        st.session_state.settings = {
            "llm_model": "gpt-3.5-turbo",
            "retriever_type": "Ensemble",
            "k": 4,
            "use_reranker": True,
            "top_n":4,
        }

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
