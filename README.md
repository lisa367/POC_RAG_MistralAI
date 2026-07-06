# Assistant virtuel pour la recommendation d'évènements culturels

Ce projet implémente un chatbot capable de recommander des événements en région parisienne pour le compte de l'entreprise Plus-Events. 
Cela est rendu possible grace à l'intégration de données contextuelles provenant d'Openagenda à un modèle IA. 
Le chatbot utilise les techonologies :
   - LangChain (gestion open source des llm)
   - FAISS (indexation des événements)
   - MistralAI (embeddings et génération de réponses).

## Structure du Projet

      ├── chatbot/
         ├── openagenda/ 
            ├── evenements-publics-openagenda.py

         ├── index_faiss/ 
            ├── index.faiss
            ├── index.pkl

         ├── vectorization.py   
         ├── chatbot.py           
         ├── tests.py
         └── streamlit_interface.py



## Pré-requis

Avant d'exécuter les scripts, assurez-vous :

- d'avoir généré une clé d'API MistralAI
- d'avoir Python 3 d'installé sur votre machine
- d'avoir créé un fichier .env dans lequel vous sauvegarderez votre clé d'API de la manière suivante : <br > MISTRAL_API_KEY=<votre-clé-api>

## Installation des dépendances

1. Créez et activez un environnement virtuel dans voter terminal:
      ```bash
      python3 -m venv venv  # sur Linux/Mac
      source venv/bin/activate  # sur Windows : venv\Scripts\activate


2. Installez les dépendances :

   ```bash
   pip install -r requirements.txt


3. Ouvrez le chatbot dans l'interface Streamlit:

   ```bash
   streamlit run streamlit_interface.py


4. Poser des questions au chatbot, par exemple : <br>

   __ Il y a-t-il des évènements musicaux de prévus à Paris pour cet été ?__
<br>
<br>
5. Pour exécuter les tests :
   ```bash
   pytest tests.py
