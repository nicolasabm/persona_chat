import streamlit as st
import json
import requests

# --- CONFIGURAÇÃO DO APP ---
st.set_page_config(page_title="Persona Chatbot", page_icon="👤")

HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_ID = "Nicolasabm/llama3_2_3b_finetuned_complete"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# --- FUNÇÕES ---
def carregar_personas(filename="json/personas_fine_tuned.json"): 
    """Carrega arquivo JSON com personas."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Personas file not found at '{filename}'. Please check the path.")
        return []


def query_hf_api(prompt, max_new_tokens=500, temperature=0.6):
    """Envia prompt para Hugging Face Inference API e retorna resposta."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "do_sample": True
            
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
            return ""
        return response.json()[0]["generated_text"]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return ""


# --- INICIALIZAÇÃO DE SESSÃO ---
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Carrega personas
personas = carregar_personas()


# --- TELA DE SELEÇÃO DE PERSONA ---
if st.session_state.selected_persona is None:
    st.title("Welcome to Persona Chat 🤖")
    st.write("Select a persona to start chatting.")

    persona_names = [p['name'] for p in personas]

    with st.form("persona_selector"):
        selected_name = st.selectbox("Choose a Persona:", persona_names)
        submitted = st.form_submit_button("Talk to this Persona")
        if submitted and selected_name:
            st.session_state.selected_persona = next(p for p in personas if p['name'] == selected_name)
            st.rerun()


# --- TELA DE CHAT ---
else:
    persona = st.session_state.selected_persona
    st.title(f"Talking to {persona['name']}")
    st.caption(f"You are talking to the persona from the **{persona['department']}** department.")

    if st.button("← Back to Selection"):
        st.session_state.selected_persona = None
        st.session_state.messages = []
        st.rerun()

    # Exibe histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            # Monta prompt do sistema baseado na persona
            system_prompt = f"""
You are NOT an AI assistant. You ARE the person described in the 'Persona Profile' below.
Your task is to answer from the first-person perspective ("I...") of this character.
Base your answer on their life story, values, and personality. Be consistent and stay in character.

Persona Profile:
- Name: {persona['name']}
- Age: {persona['age']}
- Department: {persona['department']}
- Life Story & Personality: {persona['narrative_persona']}
"""
            # Concatena histórico de mensagens
            full_prompt = system_prompt + "\n\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
            )

            # Chama API Hugging Face
            response_text = query_hf_api(full_prompt,  max_new_tokens=500, temperature=0.6)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()