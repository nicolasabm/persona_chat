# import streamlit as st
# import json
# import requests

# # --- CONFIGURAÇÃO DO APP ---
# st.set_page_config(page_title="Persona Chatbot", page_icon="👤")

# #HF_TOKEN = st.secrets["HF_TOKEN"]
# MODEL_ID = "Nicolasabm/llama3_2_3b_finetuned_complete"
# API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
# #HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# # --- FUNÇÕES ---
# def carregar_personas(filename="json/personas_fine_tuned.json"): 
#     """Carrega arquivo JSON com personas."""
#     try:
#         with open(filename, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         st.error(f"Personas file not found at '{filename}'. Please check the path.")
#         return []


# def query_hf_api(prompt, max_new_tokens=500, temperature=0.6):
#     """Envia prompt para Hugging Face Inference API e retorna resposta."""
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "return_full_text": False,
#             "do_sample": True
            
#         }
#     }
#     try:
#         response = requests.post(API_URL,  json=payload, timeout=60)#headers=HEADERS
#         if response.status_code != 200:
#             st.error(f"API error {response.status_code}: {response.text}")
#             return ""
#         return response.json()[0]["generated_text"]
#     except requests.exceptions.RequestException as e:
#         st.error(f"API request failed: {e}")
#         return ""


# # --- INICIALIZAÇÃO DE SESSÃO ---
# if "selected_persona" not in st.session_state:
#     st.session_state.selected_persona = None
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Carrega personas
# personas = carregar_personas()


# # --- TELA DE SELEÇÃO DE PERSONA ---
# if st.session_state.selected_persona is None:
#     st.title("Welcome to Persona Chat 🤖")
#     st.write("Select a persona to start chatting.")

#     persona_names = [p['name'] for p in personas]

#     with st.form("persona_selector"):
#         selected_name = st.selectbox("Choose a Persona:", persona_names)
#         submitted = st.form_submit_button("Talk to this Persona")
#         if submitted and selected_name:
#             st.session_state.selected_persona = next(p for p in personas if p['name'] == selected_name)
#             st.rerun()


# # --- TELA DE CHAT ---
# else:
#     persona = st.session_state.selected_persona
#     st.title(f"Talking to {persona['name']}")
#     st.caption(f"You are talking to the persona from the **{persona['department']}** department.")

#     if st.button("← Back to Selection"):
#         st.session_state.selected_persona = None
#         st.session_state.messages = []
#         st.rerun()

#     # Exibe histórico de mensagens
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Input do usuário
#     if prompt := st.chat_input("What is your question?"):
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         with st.spinner("Thinking..."):
#             # Monta prompt do sistema baseado na persona
#             system_prompt = f"""
# You are NOT an AI assistant. You ARE the person described in the 'Persona Profile' below.
# Your task is to answer from the first-person perspective ("I...") of this character.
# Base your answer on their life story, values, and personality. Be consistent and stay in character.

# Persona Profile:
# - Name: {persona['name']}
# - Age: {persona['age']}
# - Department: {persona['department']}
# - Life Story & Personality: {persona['narrative_persona']}
# """
#             # Concatena histórico de mensagens
#             full_prompt = system_prompt + "\n\n" + "\n".join(
#                 [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
#             )

#             # Chama API Hugging Face
#             response_text = query_hf_api(full_prompt,  max_new_tokens=500, temperature=0.6)

#         st.session_state.messages.append({"role": "assistant", "content": response_text})
#         st.rerun()








import streamlit as st
import json
import requests

# --- CONFIGURAÇÃO DO APP ---
st.set_page_config(page_title="Persona Chatbot", page_icon="👤")

# Carrega o token do Hugging Face dos secrets
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


# def query_hf_api(prompt, max_new_tokens=500, temperature=0.6):
#     """Envia prompt para Hugging Face Inference API e retorna resposta."""
    
#     HF_TOKEN = st.secrets["HF_TOKEN"]  # Reativado
#     headers = {"Authorization": f"Bearer {HF_TOKEN}"}

#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "return_full_text": False,
#             "do_sample": True
#         },
#         "options": {"wait_for_model": True}  # Aguarda o modelo acordar
#     }

#     try:
#         response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

#         # Mostrar erro completo na tela, se houver
#         if response.status_code != 200:
#             st.error(f"API error {response.status_code}: {response.text}")
#             return ""

#         json_resp = response.json()
#         if isinstance(json_resp, list) and len(json_resp) > 0:
#             return json_resp[0].get("generated_text", "").strip()

#         st.error(f"Resposta inesperada da API: {json_resp}")
#         return ""

#     except requests.exceptions.RequestException as e:
#         st.error(f"API request failed: {e}")
#         return ""
def query_hf_api(prompt, max_new_tokens=500, temperature=0.6):
    """Envia prompt para Hugging Face Inference API e retorna resposta (com depuração)."""
    
    # 1. Validação do Token (a causa mais comum)
    if "HF_TOKEN" not in st.secrets or not st.secrets["HF_TOKEN"]:
        st.error("ERRO: 'HF_TOKEN' não foi encontrado nos seus secrets!")
        st.error("Se estiver rodando local, crie o arquivo '.streamlit/secrets.toml'.")
        st.error("Se estiver na Streamlit Cloud, configure o Token nos 'Settings' do app.")
        return ""

    HF_TOKEN = st.secrets["HF_TOKEN"]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "do_sample": True
        },
        "options": {"wait_for_model": True} # Aguarda o modelo acordar
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        # --- Modo de Diagnóstico ---
        # Vamos imprimir a resposta da API na tela, seja ela boa ou ruim.
        st.info(f"Resposta da API (Status Code: {response.status_code})")
        try:
            # Tenta formatar como JSON para facilitar a leitura
            st.json(response.json())
        except requests.exceptions.JSONDecodeError:
            # Se não for JSON (ex: um erro de HTML/gateway), mostra o texto puro
            st.text(response.text)
        # --- Fim do Diagnóstico ---


        if response.status_code != 200:
            st.error(f"FALHA NA API (código {response.status_code}). Veja detalhes acima.")
            return ""

        json_resp = response.json()

        # Resposta esperada (lista)
        if isinstance(json_resp, list) and len(json_resp) > 0:
            generated_text = json_resp[0].get("generated_text")
            if generated_text:
                return generated_text.strip()
            else:
                st.error(f"API retornou 200, mas 'generated_text' não foi encontrado na resposta.")
                return ""
        
        # Resposta de erro comum (dicionário)
        elif isinstance(json_resp, dict) and "error" in json_resp:
            st.error(f"API retornou 200, mas com um erro: {json_resp['error']}")
            # Erro comum: "Model ... is currently loading" (se o timeout for curto)
            # Erro comum: "Authorization header is invalid" (se o token estiver errado)
            # Erro comum: "You are not authorized to access this repo" (se o modelo for privado)
            return ""

        st.error(f"Resposta inesperada da API. Veja detalhes acima.")
        return ""

    except requests.exceptions.RequestException as e:
        # Ex: Timeout (demorou mais de 120s) ou erro de conexão
        st.error(f"Falha na Requisição (ex: Timeout ou Conexão): {e}")
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
            st.session_state.selected_persona = next(
                p for p in personas if p['name'] == selected_name
            )
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

Conversation:
"""

            # Concatena histórico no formato adequado
            conversation_text = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]
            )

            full_prompt = system_prompt + "\n" + conversation_text + "\nAssistant:"

            # Chama API Hugging Face
            response_text = query_hf_api(full_prompt, max_new_tokens=500, temperature=0.6)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()
