from pathlib import Path
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st

from agents.generation_agent import get_hugface_model
from agents.transcript_agent import VoiceToText
from agents.video_agent import SmolVLM2ChatModel
from agents.memory import MemoryManager
from agents.langgraph_agents import build_graph

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="🎬 Short Video Analysis",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- INIT MEMORY MANAGER ----
MEMORY = MemoryManager(persist=True)
USER_ID = "user1"  # You can later change this per-login user

# ---- SESSION STATE ----
if "current_session" not in st.session_state:
    st.session_state.current_session = None

# ---- FUNCTIONS ----
def start_new_session():
    session_name = f"Session {len(MEMORY.list_sessions(USER_ID)) + 1}"
    MEMORY.add_chat_session(USER_ID, session_name)
    st.session_state.current_session = session_name

def get_current_messages():
    if st.session_state.current_session:
        return MEMORY.get_history(USER_ID, st.session_state.current_session)
    return []

def add_message(role, content):
    if st.session_state.current_session:
        MEMORY.add_message(USER_ID, st.session_state.current_session, role, content)

# ---- CACHE MODELS ----
@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache all heavy AI models."""
    hug_llm = get_hugface_model()
    transcript_model = VoiceToText()
    vlm = SmolVLM2ChatModel(model_size="medium", quantization="4bit")
    ai_workflow = build_graph()
    return hug_llm, transcript_model, vlm, ai_workflow


# ---- MAIN CONTENT ----
st.title("🎬 Short Video Analysis Chat")
st.caption("Upload a short video and let the AI analyze it scene-by-scene or frame-by-frame.")

with st.spinner("🔄 Loading Local Models..."):
    hug_llm, transcript_model, vlm, ai_workflow = load_models()

# ---- SIDEBAR ----
with st.sidebar:
    st.title("💬 Chat History")

    sessions = MEMORY.list_sessions(USER_ID)
    if not sessions:
        st.info("No sessions yet.")
    else:
        for session_name in sessions:
            if st.button(session_name, use_container_width=True):
                st.session_state.current_session = session_name
                st.rerun()

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        start_new_session()
        st.rerun()

    if st.button("🗑️ Clear All History", use_container_width=True):
        MEMORY.remove_user(USER_ID)
        st.session_state.current_session = None
        st.rerun()

# ---- MAIN LOGIC ----
if not st.session_state.current_session:
    st.info("👈 Select or create a chat session from the sidebar to begin.")
else:
    # Create 'uploaded' folder in the same directory if it doesn't exist
    upload_dir = Path(__file__).parent / "uploaded"
    upload_dir.mkdir(exist_ok=True)

    uploaded_file = st.file_uploader("📤 Upload a short video (MP4)", type=["mp4"])
    if uploaded_file:
        video_file = upload_dir / uploaded_file.name

        # Save the file contents
        with open(video_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File saved successfully at: {video_file}")
        st.write(f"Full path: `{video_file.resolve()}`")

        st.video(video_file)

        video_file = str(video_file.resolve())
        print(video_file)


    chat_container = st.container()
    messages = get_current_messages()

    # Display chat history
    if len(messages) > 0:
        with chat_container:
            for msg in messages:
                st.chat_message(msg["role"]).markdown(msg["content"])

    # User input
    user_input = st.chat_input("💭 Describe what you want to analyze...")

    if user_input and uploaded_file:
        #add_message("user", user_input)
        with chat_container:
            st.chat_message("user").markdown(user_input)

        workflow_state = {
            "user_id": USER_ID,
            "session_id": st.session_state.current_session,
            "user_query": user_input,
            "video_path": rf"{video_file}",
            "report_path": "NA",
            "hug_llm": hug_llm,
            "transcript_model": transcript_model,
            "vlm": vlm,
            "memory": MEMORY
        }

        with st.spinner(text="AI is working hard for you...", show_time=True):
            response_data = ai_workflow.invoke(
                workflow_state
            )

        final_response = response_data["final_response"]

        with chat_container:
            with st.chat_message("assistant"):
                if response_data["report_path"] == "NA":
                    st.text(final_response)
                else:
                    st.text(final_response)

                    report_file = response_data["report_path"] 
                    report_file = Path(report_file)

                    with open(report_file, "rb") as f:
                        bytes = f.read()

                    if report_file.suffix == ".pdf":
                        st.download_button(
                            label="📄 Download PDF",
                            data=bytes,
                            file_name=report_file.name,
                            mime="application/pdf"
                        )
                    elif report_file.suffix == ".pptx":
                        st.download_button(
                            label="📄 Download PPTX",
                            data=bytes,
                            file_name=report_file.name,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        )
    else:
        st.toast("Please upload a video first")

        #add_message("assistant", ai_text)

st.markdown("---")
st.caption("🧠 Built with Streamlit • Qwen3 • SmolVLM2 • Whisper • 2025 Edition")
