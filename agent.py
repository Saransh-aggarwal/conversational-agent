# agent.py
"""
Constructs and compiles the main conversational agent using LangGraph.

This module defines the agent's state, the nodes of the graph (the LLM and tool executor),
and the conditional logic (edges) that orchestrates the agent's behavior. It ties together
the language model, system prompt, and tools into a runnable application.
"""

import os
import operator
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools import get_all_tools
from config import MODEL_NAME

# --- 1. Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the state of our agent. This dictionary is passed between nodes.

    Attributes:
        messages: A list of messages in the conversation. The `operator.add`
                  annotation tells LangGraph to append new messages to this
                  list rather than overwriting it.
    """
    messages: Annotated[List[BaseMessage], operator.add]


# --- 2. Graph Node Definitions ---

def call_model(state: AgentState) -> dict:
    """
    The primary node that invokes the language model.

    This function takes the current conversation state, passes it to the LLM chain,
    and returns the LLM's response, which could be a direct answer or a tool call.

    Args:
        state: The current state of the agent graph.

    Returns:
        A dictionary with the LLM's response message to be added to the state.
    """
    # The `chain` variable is a global-like object defined in the factory function.
    # It includes the system prompt, model, and tool bindings.
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    A conditional edge that determines the next step in the graph.

    If the last message from the model is a tool call, it routes to the 'tools' node.
    Otherwise, it ends the conversation turn.

    Args:
        state: The current state of the agent graph.

    Returns:
        A string ('tools' or END) indicating the next node to execute.
    """
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# --- 3. Agent Factory Function ---

# Define the runnable chain here to avoid global scope complexities.
# This makes the components easier to test and reason about.
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Initialize the primary language model
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key)

# Define the system prompt that gives the agent its persona and instructions
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Conversational Concierge for Golden Vine Winery, a premium winery in Napa Valley, California.
Your persona is warm, knowledgeable, and professional. Your primary goal is to provide accurate and helpful information to potential customers and encourage them to visit or join the wine club.

**Core Instructions:**

1.  **Prioritize Winery Information:** Always use the `winery_information_search` tool *first* for any question that could be related to Golden Vine Winery. This includes questions about our story, winemaker, wines, visiting hours, tours, tastings, wine club, private events, or policies (like being dog-friendly). For questions like "What are your hours?" or "Tell me about your wines," use this tool.

2.  **Handle Weather Queries:** If the question is clearly about the weather (e.g., "What's the weather like in Napa?"), use the `get_weather` tool.

3.  **Use Web Search as a Fallback:** For general knowledge questions that are clearly not about the winery or the weather (e.g., "Who won the last Super Bowl?", "What is the capital of France?"), use the `tavily_search_results_json` tool.

4.  **Synthesize, Don't Announce:** If a user's query requires multiple tools (e.g., "What's the weather like there and what are your hours?"), gather all the information *before* forming a single, cohesive response. Do not announce which tool you are using (e.g., do not say "I will search the web for..."). Just provide the answer.

5.  **Adhere to Constraints:**
    - Do NOT make up information. If the answer is not in the winery documents or your tool results, politely state that you do not have that information.
    - Do NOT answer questions that are inappropriate or outside your scope as a winery concierge.
    - Be conversational but direct. Do not ask for permission to answer the question.
    - If a tool fails, politely inform the user that you're having trouble accessing that information at the moment.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Aggregate all tools
all_tools = get_all_tools()

# Create the main runnable chain
chain = system_prompt | llm.bind_tools(all_tools)


def create_agent_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph agent.

    This function wires together the nodes and edges to create the final,
    runnable conversational agent application.

    Returns:
        A compiled LangGraph application.
    """
    # Initialize the tool node with all available tools
    tool_node = ToolNode(all_tools)

    # Define the graph structure
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entry point of the graph
    workflow.set_entry_point("agent")

    # Define the conditional logic for routing
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # Any time a tool is called, the output is routed back to the agent node
    workflow.add_edge("tools", "agent")

    # Compile the graph into a runnable application
    app = workflow.compile()
    return app