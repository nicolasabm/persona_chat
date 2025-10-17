import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


#BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
#ADAPTER_PATH = "models/llama_fine_tuned_Security_Seeker"
MODEL_PATH = "Nicolasabm/llama3_2_3b_finetuned_complete"
# --- FUNÇÕES DE LÓGICA (BACKEND) ---

# Usa o cache do Streamlit para carregar o modelo apenas uma vez
@st.cache_resource
# def load_model_and_tokenizer():
#     """Carrega o modelo base, o tokenizer e aplica o adaptador PEFT."""
#     st.info("Loading base model... This may take a few minutes. 🤖")
    
#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_ID,
#         load_in_4bit=True,
#         dtype=torch.bfloat16,
#         device_map="auto",
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     # Aplica o adaptador
#     model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

#     st.success("Model loaded successfully!")
#     return model, tokenizer

def load_model_and_tokenizer():
    """Carrega o modelo completo já fundido e o tokenizer."""
    st.info("Loading merged fine-tuned model... This may take a few minutes. 🤖")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    st.success("Model loaded successfully!")
    return model, tokenizer



def carregar_personas(filename="json/personas_fine_tuned.json"): 
    """Loading personas JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Personas file not found at '{filename}'. Please check the path.")
        return []


# --- STREAMLIT APPLICATION LOGIC (FRONTEND) ---

# Configuração da página (título, ícone)
st.set_page_config(page_title="Persona Chatbot", page_icon="👤")

# Carrega o modelo e as personas
model, tokenizer = load_model_and_tokenizer()
personas = carregar_personas()

# Inicializa o estado da sessão se ainda não existir
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# --- TELA DE SELEÇÃO DE PERSONA ---
if st.session_state.selected_persona is None:
    st.title("Welcome to Persona Chat 🤖")
    st.write("Select a persona to start chatting.")
    
    # Cria uma lista de nomes de personas para o selectbox
    persona_names = [p['name'] for p in personas]
    
    # Usa um formulário para a seleção, o que parece mais organizado
    with st.form("persona_selector"):
        selected_name = st.selectbox("Choose a Persona:", persona_names)
        submitted = st.form_submit_button("Talk to this Persona")
        
        if submitted and selected_name:
            # Encontra o dicionário completo da persona selecionada
            st.session_state.selected_persona = next(p for p in personas if p['name'] == selected_name)
            # Reinicia o app para ir para a tela de chat
            st.rerun()

# --- TELA DE CHAT ---

# else:
#     persona = st.session_state.selected_persona
#     st.title(f"Talking to {persona['name']}")
#     st.caption(f"You are talking to the persona from the **{persona['department']}** department.")

#     if st.button("← Back to Selection"):
#         st.session_state.selected_persona = None
#         st.session_state.messages = []
#         st.rerun()

#     # STEP 1: This part is responsible for DRAWING the chat on the screen
#     # It reads the history and draws EACH message.
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # STEP 2: This part takes care of the LOGIC only, without drawing anything directly.
#     if prompt := st.chat_input("What is your question?"):
#         # Adds the user's message to the state (without drawing)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Generates the AI response
#         with st.spinner("Thinking..."):
#             #conversation_history = ""
#             #for message in st.session_state.messages[:-1]: # Pega todas as mensagens, exceto a última (a atual)
#             #    role = "User" if message["role"] == "user" else persona['name']
#             #    conversation_history += f"{role}: {message['content']}\n"

#             chat_prompt_text = f"""
# Role:
# You are NOT an AI assistant. You ARE {persona['name']}. Answer from their first-person perspective.
# Persona Profile:
# - Name: {persona['name']}
# - Age: {persona['age']}
# - Department: {persona['department']}
# - Life Story & Personality: {persona['narrative_persona']}

## --- CONVERSATION HISTORY ---
## {conversation_history}
## --- END OF HISTORY ---

# User Question:
# {prompt}

# Answer as {persona['name']}:
# """
#             inputs = tokenizer(chat_prompt_text, return_tensors="pt").to(model.device)
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=500,
#                 temperature=0.6,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#             response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

#         # Adiciona a resposta do assistente ao estado (sem desenhar)
#         st.session_state.messages.append({"role": "assistant", "content": response_text})

#         # ETAPA 3: Força a re-execução do script. Agora o loop da ETAPA 1
#         # vai desenhar as novas mensagens que acabamos de adicionar.
#         st.rerun()

else:
    persona = st.session_state.selected_persona
    st.title(f"Talking to {persona['name']}")
    st.caption(f"You are talking to the persona from the **{persona['department']}** department.")

    if st.button("← Back to Selection"):
        st.session_state.selected_persona = None
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):

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

            # 2. Construa a lista de mensagens no formato que o template espera.
            # A estrutura é uma lista de dicionários.
            messages_for_template = [
                {"role": "system", "content": system_prompt}
            ]
            # Adiciona todo o histórico da conversa (usuário e assistente)
            messages_for_template.extend(st.session_state.messages)
            
            # 3. Use o tokenizer para aplicar o template de chat oficial do Llama 3.
            # Isso converte a lista de mensagens em uma única string formatada corretamente.
            # `add_generation_prompt=True` adiciona os tokens que sinalizam ao modelo que ele deve começar a gerar uma resposta.
            final_prompt_string = tokenizer.apply_chat_template(
                messages_for_template, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 4. Tokenize o prompt final e gere a resposta.
            inputs = tokenizer(final_prompt_string, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()