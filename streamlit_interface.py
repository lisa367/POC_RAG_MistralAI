import streamlit as st
from datetime import datetime, date
from chatbot import get_chatbot_answer


"""
Script to create and set configuration of streamlit interface for our chatbot.
"""

if "location" not in st.session_state:
    st.session_state.location = "Paris"

st.set_page_config(page_title="Chatbot EventParigo", page_icon="🤖")
st.title("EventParigo - Le Chatbot spécialiste des évènements sur Paris 🥂📅")
st.markdown("Demandez des informations ou des recommendations sur les événements à venir en Ile-de-France")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": """
            Bonjour, je suis un assistant virtuel spécialisé dans l'évennementiel en région parisienne.
            Demandez-moi des informations :)"""
    }]

user_input = st.chat_input("Posez votre question ici...")

# Traitement de la question et réponse
if user_input:
    location = st.session_state.location
    response = get_chatbot_answer(user_input)
    st.session_state.chat_history.append(("Vous", user_input))
    st.session_state.chat_history.append(("Assistant", response))

# Afficher l’historique du chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role.lower()):
        st.markdown(msg)