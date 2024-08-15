import streamlit as st
import requests

# Set the title of the app
st.set_page_config(page_title="Medical ChatBot", page_icon="💊")

# Display a title and description
st.title("Medical ChatBot")
st.write("Ask me questions about motor neuron disease.")

# Create a sidebar for additional information or navigation
st.sidebar.header("About")
st.sidebar.write("This chatbot helps you with questions about motor neuron disease.\nJust type your question and click 'Ask Questions'.")

# User input
user_input = st.text_input("Type your question here:")

if st.button("Ask Questions"):
    if user_input:
        with st.spinner("Fetching response..."):
            # Make a request to the selected endpoint
            response = requests.post(
                "http://fastapi:8000/chat",
                json={"message": user_input}
            )
            
            if response.status_code == 200:
                data = response.json()

                # Display the generated text in a text area
                st.text_area("Generated Response", data["response"], height=300, max_chars=1000)
   
            else:
                st.error("Failed to get a response. Please try again.")
    else:
        st.warning("Please enter a question before clicking 'Ask Questions'.")
