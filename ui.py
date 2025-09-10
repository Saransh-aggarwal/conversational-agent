# ui.py
"""
Provides a web-based user interface for the Conversational Concierge using Streamlit.

This script creates an interactive chat application that allows users to communicate
with the agent. It manages chat history, displays the conversation, and handles
the streaming of the agent's responses in real-time.
"""

import asyncio
import platform
import streamlit as st

# --- CRITICAL: ASYNCIO EVENT LOOP FIX ---
# This must be the first part of the script, before any other project imports.
# Streamlit runs in a different thread on Windows, which lacks a default asyncio
# event loop. This creates and sets an event loop for the current thread.
if platform.system() == "Windows":
    asyncio.set_event_loop(asyncio.new_event_loop())
# -----------------------------------------

# Now it is safe to import our other modules
from langchain_core.messages import AIMessage, HumanMessage
from agent import create_agent_graph


# --- Page Configuration (Should be the first Streamlit command) ---
st.set_page_config(
    page_title="Conversational Concierge",
    page_icon="🍇",
    layout="centered",
)

st.title("Golden Vine Winery Conversational Concierge 🍇")


# --- State Management & Agent Initialization ---
# Initialize agent and chat history in Streamlit's session state to persist
# them across user interactions (reruns).

# Initialize the agent graph only once.
if "agent_app" not in st.session_state:
    with st.spinner("Initializing Agent... This may take a moment."):
        try:
            st.session_state.agent_app = create_agent_graph()
        except Exception as e:
            st.error(f"Failed to initialize the agent. Please check your API keys and configuration. Error: {e}")
            # Stop the script if the agent can't be created.
            st.stop()

# Initialize the chat message history if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(
            content="Welcome! How can I assist you with Golden Vine Winery, the weather, or anything else?"
        )
    ]


# --- Chat History Display ---
# Display the existing messages in the chat history.
for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)


# --- User Input Handling ---
# The st.chat_input widget waits for and captures the user's input.
if user_prompt := st.chat_input("Ask me anything..."):
    # Add the user's message to history and display it immediately.
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # --- Agent Response Streaming ---
    with st.chat_message("assistant"):
        # Use an empty placeholder to stream the agent's response chunk by chunk.
        response_placeholder = st.empty()
        full_response = ""

        # Prepare the input for the agent graph.
        inputs = {"messages": [HumanMessage(content=user_prompt)]}
        
        # The spinner provides a visual cue that the agent is working.
        with st.spinner("Thinking..."):
            # Stream events from the agent graph.
            for event in st.session_state.agent_app.stream(inputs):
                if "agent" in event:
                    message_chunk = event["agent"]["messages"][-1]
                    if message_chunk.content:
                        full_response += message_chunk.content
                        # Update the placeholder with the latest content and a typing cursor.
                        response_placeholder.markdown(full_response + "▌")
        
        # Display the final, complete response without the typing cursor.
        response_placeholder.markdown(full_response)

    # Add the final AI response to the chat history for the next turn.
    st.session_state.messages.append(AIMessage(content=full_response))