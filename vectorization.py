import os
import re
import pickle
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from tqdm import tqdm


load_dotenv()
API_KEY = os.environ.get("MISRAL_API_KEY")

if not API_KEY:
    raise Exception("MissingApiKeyError: You need to provide your API key to proceed.")

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=API_KEY)

def html_to_text(x: pd.Series) -> pd.Series:
    """
    Converts html to str.

    Parameters:
        x: pd.Series

    Returns:
        x: pd.Series
    """
    for col in ["title", "description", "longdescription"]:
        text = BeautifulSoup(x[col], "html.parser").get_text()
        text = re.sub(r'[^\w\s.,!?;:\'\"À-ÿ]', ' ', text) # Removing some special characters and accents
        text = ' '.join(text.split())
        x[col] = text.lower()
    return x

def create_content(x: pd.Series) -> str:
    """
    Creates content from event data.

    Parameters:
        x: pd.Series

    Returns:
        str
    """
    return (
        f"Titre: {x.title}\n"
        f"Description: {x.description}\n"
        f"Lieu: {x.location_name}, {x.location_address}, {x.location_postalcode} {x.location_city}\n"
        f"Dates: {x.firstdate_begin}, {x.lastdate_end}"
    )

def create_metadata(x: pd.Series) -> dict:
    """
    Creates metadata from event data.

    Parameters:
        x: pd.Series

    Returns:
        dict
    """
    return {
            "source": "openagenda",
            "id": x.uid,
            "title": x.title,
            "description": x.description,
            "firstdate_begin": x.firstdate_begin,
            "lastdate_end": x.lastdate_end,
            # "date_fin": x.date_fin,
            "location_name": x.location_name,
            "location_address": x.location_address,
            "location_district": x.location_district,
            "location_postalcode": x.location_postalcode,
            "location_city": x.location_city,
            "location_description": x.location_description,
            "location_coordinates": x.location_coordinates,
            "keywords": x.keywords,
        }

def clean_events_data(data_path: str) -> pd.DataFrame:
    """
    Cleans raw data events from json file.

    Parameters:
        data_path: str

    Returns:
        df_events_one_year: pd.DataFrame
    """

    # Importing and cleaning events
    df_events = pd.read_json(data_path)
    df_events.dropna(subset=['title_fr', 'description_fr', 'uid'])
    df_events.drop_duplicates(subset="uid", inplace=True)
    cols_to_rename = {col: col[:-3] for col in df_events.columns if col.endswith("_fr")}
    df_events = df_events.rename(cols_to_rename, axis=1)
    df_events = df_events.fillna({"title": "", "description": "", "longdescription": "", "location_name": "", "location_address": "", 'location_description': "", "location_postalcode": 0, "location_city": "", "country": ""})

    df_events = df_events.apply(html_to_text, axis=1)

    # Filtering on events less than a year old
    df_events["firstdate_begin"] = pd.to_datetime(df_events["firstdate_begin"], errors="coerce", utc=True)
    df_events["lastdate_end"] = pd.to_datetime(df_events["lastdate_end"], errors="coerce", utc=True)
    df_events_one_year = df_events.loc[df_events.firstdate_begin > (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365))].copy()
    df_events_one_year = df_events_one_year.astype({"firstdate_begin": "str", "lastdate_end": "str"})

    # Creating documents
    df_events_one_year["page_content"] = df_events_one_year.apply(create_content, axis=1)
    df_events_one_year["metadata"] = df_events_one_year.apply(create_metadata, axis=1)
    df_events_one_year["document"] = df_events_one_year.apply(lambda x: Document(page_content=x.page_content, metadata=x.metadata), axis=1)

    return df_events_one_year

def df_to_documents(df: pd.DataFrame):
    """
    Retrieve documents of events from a dataframe.

    Parameters:
        df: pd.DataFrame

    Returns:
        list of documents : list[Document]
    """
    return df.document.tolist()

def split_documents(documents):
    """
    Retrieve documents of events from a dataframe.

    Parameters:
        documents: list[Document]

    Returns:
        list of chunks of documents
    """
    text_splitter = SemanticChunker(embeddings)
    chunks = []
    for document in tqdm(documents, desc="Découpage des documents", unit="document"):
        chunks.extend(text_splitter.create_documents(texts=[document.page_content], metadatas=[document.metadata]))
    return chunks

def run_vectorization(data_path):
    """
    Executes the vectorization tasks.
    """
    events = clean_events_data(data_path)
    documents = df_to_documents(events)
    docs_chunked = split_documents(documents)
    vectorstore = FAISS.from_documents(docs_chunked, embeddings)
    vectorstore.save_local("index_faiss")

    # Test :
    query = "événements de musique à Paris"
    print(f"Recherche sémantique : {query}")
    docs_retrieved = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(docs_retrieved, 1):
        print(f"\n Résultat {i}:")
        print(doc.page_content)
        print(" Métadonnées:", doc.metadata)
    return

if __name__ == "__main__":
    run_vectorization("openagenda/evenements-publics-openagenda.json")
