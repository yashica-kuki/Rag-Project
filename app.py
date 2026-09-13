import streamlit as st
import requests

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("🤖 Hackathon RAG Assistant")

# 1. Initialize chat history in the browser
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Redraw chat messages on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new user input
if user_input := st.chat_input("Ask a question about the rulebooks..."):
    # Display user message instantly
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 4. Call the FastAPI Backend
    with st.spinner("Searching documents..."):
        try:
            res = requests.post("http://localhost:8000/chat", json={"question": user_input})
            res.raise_for_status() # Check for errors
            data = res.json()
            
            # Format the AI's response and attach the citations
            answer = data["answer"]
            source_text = "\n\n**Sources:**\n"
            for s in data["sources"]:
                source_text += f"- *Page {s['page']}*: {s['content']}\n"
                
            full_response = answer + source_text
            
            # Display the AI response
            with st.chat_message("assistant"):
                st.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to the backend. Is FastAPI running?")