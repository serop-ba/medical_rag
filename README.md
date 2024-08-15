Medical_Rag
==============================

This repository uses ollama server to generate embeddings for PDF documents locally which then can be used in a chatbot. 

- The chatbot UI is written in streamlit. 

- The backend is written in fastapi. 

- Langchain is used to call ollama and generate the embeddings and create the RAG application.


# Setup
You need to follow these steps:
1. Start the ollama server locally:
Either run 

```
docker-compose up --build -d 
```
Which will create all three micro services for you

or 

```
docker build -t ollama . 

docker run --publish 11434:11434 -d  ollama 
```
This will start a docker container with the ollama server running inside. 

2. Generate embeddings 

use the script under ./src/models/train_models.py

Here you specify the file you want to generate embeddings from. 

Currently we are only supporting PDF files. 


You can install the requirements using 

```
pip install -r requirements.txt

```
you can start the script using 
```
python src/models/train_model.py
```
3. Test the chatbot 

you can use the script in ./src/models/predict_model.py after you add your file

```
python src/models/train_model.py
```
This will generate the answer from the chatbot

4. If you want to have the full app
you can start the app using the following command

Before you build the app, you need to the copy the procceed embeddings to the app.

I have a data folder with the proceeded embeddings and raw 
```
docker-compose up --build -d 
```
----
This will create
- streamlit app on: http://0.0.0.0:8501
-  fastapi server on: http://0.0.0.0:8000
- ollama server on: http://0.0.0.0:11434 






