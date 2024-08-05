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

# Switch button to select between local and server endpoints
# use_local = st.radio(
#     "Choose endpoint",
#     ("Local_LLM", "Claude_Anthropic"),
#     index=0,
#     help="Select 'Local' to use the local endpoint or 'Server' to use the server endpoint."
# )

# Set the API endpoint based on the switch
# endpoint = "http://fastapi:8000/chat" if use_local == "Local_LLM" else "http://fastapi:8000/chat_server"

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
                
                # Add a button to copy the text to clipboard
                # st.download_button(
                #     label="Copy Response",
                #     data=data["response"],
                #     file_name="response.txt",
                #     mime="text/plain"
                # )
            else:
                st.error("Failed to get a response. Please try again.")
    else:
        st.warning("Please enter a question before clicking 'Ask Questions'.")
