"""
streamlit_app.py: Production-grade Streamlit Cloud & Google Cloud Run UI 
for the Interactive ESL AI Language Tutor & Slideshow app with n8n Webhook integration.
"""

import os
import json
import requests
import streamlit as st
from api import get_langgraph_app

# Set Page Config
st.set_page_config(
    page_title="Speakology ESL AI Tutor",
    page_icon="🗣️",
    layout="wide"
)

# ------------------------------------------------------------------------------
# Helper Functions: Safe Secret & Webhook Resolution
# ------------------------------------------------------------------------------
def resolve_openai_api_key() -> str:
    """
    Safely resolves the OpenAI API key across Streamlit Secrets, 
    OS Environment Variables, and User Input without raising StreamlitSecretNotFoundError.
    """
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key

    return st.session_state.get("ui_openai_api_key", "")


def resolve_n8n_webhook_url() -> str:
    """
    Safely resolves the N8N Webhook URL across Streamlit Secrets,
    OS Environment Variables, and User Input.
    """
    try:
        if "N8N_WEBHOOK_URL" in st.secrets:
            return st.secrets["N8N_WEBHOOK_URL"]
    except Exception:
        pass

    env_url = os.getenv("N8N_WEBHOOK_URL", "")
    if env_url:
        return env_url

    return st.session_state.get("ui_n8n_webhook_url", "")


def trigger_n8n_webhook(webhook_url: str, payload: dict) -> dict:
    """
    Sends event payload to n8n automation workflow via HTTP POST.
    """
    if not webhook_url:
        return {"status": "skipped", "reason": "No Webhook URL provided"}
    
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "success", "data": response.json() if response.content else {}}
        return {"status": "error", "code": response.status_code}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# ------------------------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Lesson & API Setup")
    
    # OpenAI API Key Handling
    resolved_key = resolve_openai_api_key()
    if not resolved_key:
        st.warning("No OpenAI API Key detected.")
        user_key = st.text_input("Enter OpenAI API Key:", type="password", key="ui_key_input")
        if user_key:
            st.session_state["ui_openai_api_key"] = user_key
            resolved_key = user_key
    else:
        st.success("OpenAI API Key connected!", icon="✅")

    st.divider()

    # n8n Webhook Configuration
    n8n_url = resolve_n8n_webhook_url()
    st.subheader("🔗 n8n Automation Webhook")
    if not n8n_url:
        user_n8n = st.text_input(
            "Enter n8n Webhook URL (Optional):", 
            placeholder="https://n8n.yourdomain.com/webhook/...",
            key="ui_n8n_input"
        )
        if user_n8n:
            st.session_state["ui_n8n_webhook_url"] = user_n8n
            n8n_url = user_n8n
    else:
        st.success("n8n Webhook Active!", icon="⚡")

    st.divider()

    # Curriculum & Slide Controls
    proficiency_level = st.selectbox(
        "Target CEFR Level",
        ["A1 - Beginner", "A2 - Elementary", "B1 - Intermediate", "B2 - Upper Intermediate"],
        index=1
    )
    
    slide_num = st.slider("Select Slide Step", min_value=1, max_value=3, value=1)
    
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------------------
# Curriculum Slides Database
# ------------------------------------------------------------------------------
slides_db = {
    1: {
        "title": "Ordering at a Café",
        "target": "Grammar: 'I would like...' / Vocabulary: Coffee, Drinks",
        "scaffolding_hint": "Prompt the user to ask for items politely.",
        "image": "☕ Café Menu: Espresso ($3), Latte ($4), Croissant ($5)"
    },
    2: {
        "title": "Asking for Directions",
        "target": "Grammar: Prepositions of place / Vocabulary: Straight, Left, Right",
        "scaffolding_hint": "Ask where the train station or park is located.",
        "image": "🗺️ City Map: Museum (North), Station (East), Hotel (West)"
    },
    3: {
        "title": "Describing Your Weekend",
        "target": "Grammar: Past Simple tense ('went', 'saw', 'cooked')",
        "scaffolding_hint": "Encourage past tense usage and expand on detail.",
        "image": "🏖️ Weekend Activities: Beach, Cinema, Cooking at home"
    }
}

current_slide = slides_db.get(slide_num, slides_db[1])

# ------------------------------------------------------------------------------
# Main Layout
# ------------------------------------------------------------------------------
st.title("🗣️ Speakology ESL AI Tutor")
st.caption("Interactive AI-Powered Spoken Language Practice & Dynamic Scaffolding")

col1, col2 = st.columns([1, 1], gap="medium")

# Left Column: Slide Content Display
with col1:
    st.subheader(f"Slide {slide_num}: {current_slide['title']}")
    
    st.info(f"🎯 **Target Objective:** {current_slide['target']}")
    
    st.markdown(
        f"""
        <div style="background-color: #1e1e24; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 10px;">
            <h4>📺 Visual Context Cue</h4>
            <p style="font-size: 1.2rem;">{current_slide['image']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(f"**💡 Teacher Note:** {current_slide['scaffolding_hint']}")

# Right Column: Interactive Conversational Chat
with col2:
    st.subheader("💬 Spoken & Text Practice")
    
    # Initialize chat history
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": f"Hi! Welcome to Slide {slide_num}. Take a look at the cue on the left. How can I help you today?"
            }
        ]

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Require API Key before proceeding
    if not resolved_key:
        st.warning("⚠️ Please provide an OpenAI API Key in Streamlit Secrets, Environment Variables, or the sidebar to begin practice.")
        st.stop()

    # User Input Field
    if user_input := st.chat_input("Type your answer or spoken response..."):
        # Append and display user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate AI response via LangGraph pipeline & Trigger n8n Webhook
        with st.chat_message("assistant"):
            with st.spinner("AI Tutor is analyzing your language..."):
                try:
                    # 1. Execute LangGraph pipeline
                    graph = get_langgraph_app(resolved_key)
                    initial_state = {
                        "user_input": user_input,
                        "level": proficiency_level.split(" ")[0],
                        "slide": slide_num,
                        "rag_context": "",
                        "response": ""
                    }
                    
                    final_state = graph.invoke(initial_state)
                    bot_response = final_state["response"]

                    # 2. Trigger n8n Automation Webhook (Async/Non-blocking)
                    if n8n_url:
                        webhook_payload = {
                            "event": "esl_chat_turn",
                            "user_input": user_input,
                            "ai_response": bot_response,
                            "level": proficiency_level,
                            "slide_number": slide_num,
                            "slide_title": current_slide["title"]
                        }
                        trigger_n8n_webhook(n8n_url, webhook_payload)
                    
                except Exception as e:
                    bot_response = f"⚠️ **Execution Error:** {str(e)}"

                st.write(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})