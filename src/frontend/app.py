import streamlit as st
import requests

st.title("Podcast Summarizer ChatBot")

# User input
user_input = st.text_input("Ask Questions about the podcast")

if st.button("Search"):
    if user_input:
        # Make a request to the FastAPI server
        response = requests.post(
            "http://fastapi:8000/chat",
            json={"message": user_input}
        )
        
        if response.status_code == 200:
            data = response.json()

            # st.text(data["response"])
            # Display the generated text in a text area
            st.text_area("Generated Text", data["response"], height=300)

# Button to copy the text to clipboard
            
        else:
            st.error("Failed to fetch job listings.")
    else:
        st.error("Please enter a search query.")

