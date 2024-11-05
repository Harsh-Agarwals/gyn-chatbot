# Gynecological Conversational Chatbot

This repository contains the code base for Conversational Chatbot for Gynecological domain. The knowledge base for this chatbot includes two textbooks:
- Book on Pregnancy - a complete guide by NHS (Dept. of health, Govt. of UK)
- Textbook on Gynecology by DC Dutta

These 2 books were used as these are widely used in academia and in practice. Also the knowledge base is really wide covering wide variety of topics.

The technologies used in making this chatbot are:
- LLM Framework: LangChain
- Model: Pretrained OpenAI gpt-4o-mini
- Streamlit for frontend

Here is how to run this chatbot.

- make a new folder: gyne-bot (mkdir gyne-bot)
- getting inside the folder: cd gyne-bot
- Making a virtual environment: python -m venv venv
- activate virtual environment: venv\Scripts\activate

## Now, there are 2 ways to run this application:
1. Using command line
- go to command line
- pip install git+ssh://git@github.com/Harsh-Agarwals/gyn-chatbot.git
Now all dependencies are installed and app is ready to run
- Cloning the repository git clone git@github.com:Harsh-Agarwals/gyn-chatbot.git
- type python in cmd
- import src
- from src.helper import question_answering, from_loading_to_embedding, rag_chain_init

2. Using Streamlit
- Cloning the repository git clone git@github.com:Harsh-Agarwals/gyn-chatbot.git
- Installing dependencies: pip install -m requirements.txt
- Running streamlit application: streamlit run app.py