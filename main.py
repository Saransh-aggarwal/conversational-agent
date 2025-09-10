# main.py
"""
Provides a simple command-line interface (CLI) for interacting with the
Conversational Concierge agent.

This script initializes the agent and enters a loop to accept user input,
stream the agent's responses back to the console, and handle the exit command.
It serves as the primary entry point for running the application in a terminal.
"""

from langchain_core.messages import HumanMessage
from agent import create_agent_graph

def run_console_interface():
    """
    Initializes the agent and runs the main interactive console loop.
    """
    print("Initializing agent...")
    try:
        # Create the compiled LangGraph agent application
        app = create_agent_graph()
    except Exception as e:
        print(f"Error: Failed to initialize the agent. {e}")
        return

    print("\nWelcome to the Golden Vine Winery Conversational Concierge! 🍇")
    print("You can ask me about our winery, the weather, or anything else.")
    print("Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Concierge: Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Define the input for the agent graph
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            print("Concierge: ", end="", flush=True)
            
            # Stream the response from the agent
            full_response = ""
            for event in app.stream(inputs):
                # The event stream contains outputs from each node as it executes
                if "agent" in event:
                    # We are interested in the final output from the "agent" node
                    message = event["agent"]["messages"][-1]
                    if message.content:
                        # Print the content chunk by chunk for a streaming effect
                        print(message.content, end="", flush=True)
                        full_response += message.content
            
            # Print a newline after the complete response is streamed
            print()

        except KeyboardInterrupt:
            print("\nConcierge: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    run_console_interface()