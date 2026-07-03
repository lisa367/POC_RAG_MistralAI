import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime as dt
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()
logs_path = Path().cwd()/"logs"
logs_path.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("MISRAL_API_KEY")
# print(API_KEY)

if not API_KEY:
    raise Exception("MissingApiKeyError: You need to provide your API key to proceed.")


QUERY_PROMPT = \
"""
Tu es un assistant avec une connaissance spécialisée dans les événements en région Ile-de-France.
Tu as accès à l'historique de la conversation avec l'utilisateur.
Tiens compte de la date du jour pour ne pas proposer des évènements passés.

Voici l'historique de la conversation :
{chat_history}

Voici le Contexte :
{context}

Voici la question de l'utilisateur :
{question}

Si l'utilisateur te pose une question sur un sujet qui ne porte pas sur la recherche d'évenements, réponds uniquement à cette question sans proposer d'événements.
Si tu ne trouves pas d'information dans la mémoire ou les documents, dis-le simplement sans inventer de réponse.
Réponds toujours en français.
"""

QUERY_DATE = dt.now().strftime("%Y-%m-%d %H:%M:%S")

# chatbot_logger = logging.getLogger("ChatbotLogger")
# chatbot_logger.handlers.clear()

# file_handler = logging.FileHandler(f"logs/chatbot_{QUERY_DATE}.log", encoding="utf-8")
# file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# chatbot_logger.addHandler(file_handler)
# chatbot_logger.setLevel(logging.INFO)

logging.basicConfig(
    filename=f"logs/chatbot_{QUERY_DATE}.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=API_KEY)

def download_index(index_path: str, embeddings):
    """
    Retrieve context data (FAISS index) from specified index folder.

    Parameters:
        index_path: str
        embeddings: MistralAIEmbeddings

    Returns:
        FAISS index or None
    """
    if not Path(index_path).exists() or not (Path(index_path) / "index.faiss").exists():
        logging.error(f"Index FAISS introuvable dans : {index_path}")
        return

    try:
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logging.error(f"Erreur de chargement de l'index (FAISS) : {e}")
        return

def get_chatbot_answer(user_question: str) -> str:
    """
    Sends question to LLM model with context data.

    Parameters:
        index_user_questionpath: str

    Returns:
        str
    """
    index_path = "faiss_index_1000"
    vectorstore_downloaded = download_index(index_path, embeddings)
    if vectorstore_downloaded is None:
        raise RuntimeError("Index FAISS introuvable ou erreur lors du chargement")

    memory = ConversationBufferWindowMemory(
        k=3,
        return_messages=True,
        memory_key="chat_history"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatMistralAI(name="mistral-small", api_key=API_KEY),
        retriever=vectorstore_downloaded.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(QUERY_PROMPT),
            "document_variable_name": "context"
        }
    )
    try:
        chatbot_query = qa_chain.invoke({"question": f"{user_question} Voici la date du jour : {QUERY_DATE}"})
        answer = chatbot_query.get("answer", "").strip()
        return answer

    except Exception as e:
        logging.exception("Erreur lors de la génération de la réponse :")
        return "Une erreur est survenue lors du traitement de votre question. Veuillez ré-essayer."
    

if __name__ == "__main__":
    question = "Il y a-t-il des évènements de musique à Paris pour cet été ?"
    answer = get_chatbot_answer(question)
    print(answer)