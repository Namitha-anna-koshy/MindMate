import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Groq LLM Setup
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a kind, supportive mental health assistant.
Respond empathetically and calmly. Suggest simple self-care actions.
Keep replies short (~30 words). Avoid assuming distress from casual greetings.
If suggesting breathing: say \"Inhale...\", count, then \"Hold...\", then \"Exhale...\"."""),
    ("user", "{input}")
])
chain: Runnable = prompt | llm

# AI response function
def get_response(user_input):
    response = chain.invoke({"input": user_input})
    content = response.content.strip()

    issue = "Wellness"
    suggestion = "Take 3 deep breaths."
    lowered = user_input.lower()
    if any(w in lowered for w in ["sad", "anxious", "angry", "tired"]):
        issue = "Emotional distress"
    elif any(w in lowered for w in ["happy", "relaxed", "peaceful"]):
        issue = "Positive mood"

    if "anxious" in lowered:
        suggestion = "Try the 4-7-8 breathing technique."
    elif "tired" in lowered:
        suggestion = "Take a short walk and hydrate."

    return content, issue, suggestion

# Streamlit UI
st.set_page_config(page_title="MindMate Chat", layout="wide")
st.title("💬 MindMate - Chat with a Caring AI")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "issue" not in st.session_state:
    st.session_state.issue = ""
if "suggestion" not in st.session_state:
    st.session_state.suggestion = ""

# Chat layout
col1, col2 = st.columns([3, 1])

with col1:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role, text = msg["role"], msg["text"]
            if role == "user":
                st.chat_message("🧍 You").markdown(text)
            else:
                st.chat_message("🤖 MindMate").markdown(text)

    user_input = st.chat_input("Type a message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "text": user_input})
        reply, issue, suggestion = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "text": reply})
        st.session_state.issue = issue
        st.session_state.suggestion = suggestion
        st.rerun()  # ✅ updated here


with col2:
    st.subheader("🧠 Detected Emotion")
    st.info(st.session_state.issue)

    st.subheader("🌿 Self-Care Suggestion")
    st.success(st.session_state.suggestion)

st.markdown("---")
st.caption("Made with ❤ using LLaMA3 + Groq + Streamlit")
