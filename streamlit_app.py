"""
streamlit_app.py: Main entrypoint for Streamlit Cloud & Google Cloud Run.
Reads API key securely from st.secrets or user input.
"""
import os
import streamlit as st
from api import get_langgraph_app

st.set_page_config(page_title="Speakology ESL AI Tutor", layout="wide")

st.title("🗣️ Interactive ESL AI Tutor & Slideshow")
st.caption("AI-Powered Spoken Language Practice with Dynamic Scaffolding")

# Retrieve API key securely from Streamlit secrets or sidebar input
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    if not openai_api_key:
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    proficiency_level = st.selectbox("Proficiency Level (CEFR)", ["A1 - Beginner", "A2 - Elementary", "B1 - Intermediate", "B2 - Advanced"], index=1)
    slide_num = st.slider("Current Slide", 1, 5, 1)

if not openai_api_key:
    st.info("Please provide an OpenAI API Key in Streamlit Secrets or sidebar to start.", icon="🔑")
    st.stop()

# Slide Content Rendering
slides = {
    1: {"title": "Ordering at a Café", "target": "Phrases: 'I would like...', 'How much is...?'", "image": "☕ Café Menu: Coffee ($3), Sandwich ($6), Tea ($2)"},
    2: {"title": "Asking for Directions", "target": "Phrases: 'Where is...?', 'Turn left/right'", "image": "🗺️ City Map: Library, Station, Park"},
}

current_slide_data = slides.get(slide_num, slides[1])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Slide {slide_num}: {current_slide_data['title']}")
    st.info(f"**Target Objective:** {current_slide_data['target']}")
    st.metric(label="Visual Cue", value=current_slide_data['image'])

with col2:
    st.subheader("Interactive Voice / Text Practice")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! Welcome to Slide {slide_num}. What would you like to order today?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Type or speak your answer..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.spinner("AI Tutor is thinking..."):
            try:
                graph = get_langgraph_app(openai_api_key)
                state = {
                    "user_input": user_input,
                    "level": proficiency_level.split(" ")[0],
                    "slide": slide_num,
                    "rag_context": "",
                    "response": ""
                }
                res = graph.invoke(state)
                bot_response = res["response"]
            except Exception as e:
                bot_response = f"Error generating response: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.chat_message("assistant").write(bot_response)