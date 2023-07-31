import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 256,
        temperature = 0.2
    )
    return llm

st.title("🦙Llama2🦜CSV🦙")
st.markdown("<h3 style='color: black;'>Harness the power of LLama2 with Langchain.</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='color: black;'>Developed by <a href='https://github.com/rohan-shaw'>Rohan Shaw</a> with ❤️</h4>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("CSV file here", type="csv")

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as t:
        t.write(uploaded_file.getvalue())
        temp_path = t.name

    loader = CSVLoader(file_path=temp_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
    #st.json(data)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Bhai, " + uploaded_file.name + " is file ke bare mein kuch bhi puch le aankh 👀 band karke answer dunga 🤔"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Aur, bol kya hal chal ! 🖖"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Apne CSV file ke data ke bare me yaha pe puch (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="human")
                message(st.session_state["generated"][i], key=str(i), avatar_style="pixel-art-neutral")
