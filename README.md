# 🍇 Conversational Concierge for a Napa Valley Winery

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-LangGraph-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

A smart conversational agent designed for a fictional Napa Valley winery, "Golden Vine Winery." This agent acts as a digital concierge, capable of answering detailed questions about the business, performing real-time web searches, and providing practical updates like the weather.

The project is built with a modern, tool-based architecture using LangGraph, ensuring it is accurate, fast, and flexible.

---

## 🚀 Project Deliverables

### 📄 Project Report

A detailed report covering the project's objectives, architecture, agent design process, and outcomes.

**[➡️ View the Full Project Report (PDF)](https://github.com/Saransh-aggarwal/conversational-agent/blob/main/Conversational_Concierge_Project_Checklist.pdf)**

### 🎬 Video Demonstration

A short demonstration of the agent's ability to handle a complex, multi-part query that requires using all three of its tools in a single turn. Click the thumbnail below to watch on YouTube.

[![Project Demo Video](https://img.youtube.com/vi/UUvDzDP3rwE/hqdefault.jpg)](https://youtu.be/UUvDzDP3rwE)

---

## ✨ Key Features

-   🧠 **Deep Winery Knowledge:** Answers complex questions about the winery's history, wines, winemaker, tasting experiences, and wine club using a Retrieval-Augmented Generation (RAG) system.
-   🌐 **Real-Time Web Search:** Accesses the internet via Tavily to answer general knowledge questions about current events, facts, and anything beyond its core knowledge base.
-   🌦️ **Live Weather Updates:** Provides current weather conditions for any city, a practical tool for visitors planning a trip to Napa.
-   🤖 **Intelligent Tool Orchestration:** Built with LangGraph, the agent can seamlessly decide which tool to use, or even use multiple tools in sequence, to answer complex, multi-part questions in a single, synthesized response.
-   💬 **Interactive UI:** A simple and clean web interface built with Streamlit for an intuitive and engaging user experience.

## 🏛️ Architecture Overview

The agent operates on a modular, tool-driven architecture orchestrated by LangGraph. When a user provides input, the LangGraph agent uses LLM-powered reasoning to select the most appropriate tool (or tools) to fulfill the request. The results are then synthesized into a coherent final answer and presented to the user.

![LangGraph Agent Architecture](https://github.com/Saransh-aggarwal/conversational-agent/blob/main/architecture%20(2).png?raw=true)

1.  The **Streamlit UI** captures the user's query.
2.  The **LangGraph Agent** receives the query and, using the Gemini LLM, decides which tool is best suited to answer it.
3.  The appropriate **Tool** is executed (e.g., the RAG tool retrieves relevant text from the winery's documents).
4.  The tool's output is sent back to the agent.
5.  The **LLM synthesizes** the information into a natural, human-readable response.
6.  The final response is streamed back to the **Streamlit UI**.

## 🛠️ Tech Stack

-   **Core Framework:** LangGraph
-   **LLM:** Google Gemini (`gemini-1.5-flash-latest`)
-   **Embeddings:** Google (`models/embedding-001`)
-   **Vector Store:** FAISS (for local RAG)
-   **Tools:** Tavily (Web Search), OpenWeatherMap (Weather API)
-   **UI:** Streamlit
-   **Language:** Python 3.9+

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

-   Python 3.9 or higher
-   Git for cloning the repository

### 2. Clone the Repository

```bash
git clone https://github.com/YourUsername/conversational-concierge-agent.git
cd conversational-concierge-agent
```
*(Remember to replace `YourUsername` with your actual GitHub username!)*

### 3. Set Up the Python Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

The agent relies on several external APIs and requires API keys to function.

1.  Locate the `.env.example` file in the project root.
2.  Create a copy of this file and name it `.env`.
3.  Obtain the necessary API keys:
    -   `GOOGLE_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/app/apikey).
    -   `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com/).
    -   `OPENWEATHER_API_KEY`: Get from [OpenWeatherMap](https://openweathermap.org/api).
4.  Open your `.env` file and paste your secret keys as the values for the corresponding variables.

### 6. Running the Application

You can interact with the agent in two ways:

#### A) Via the Streamlit Web UI (Recommended)

This is the best way to experience the agent.

```bash
streamlit run ui.py
```

This command will start a local web server and open the application in a new browser tab.

#### B) Via the Command-Line Interface

For a simpler, terminal-based interaction:

```bash
python main.py
```

## 📂 Project Structure

The project is organized into modular files to separate concerns and improve maintainability.

```
.
├── faiss_index/          # Stores the pre-computed vector index for RAG
├── .env                  # (Local) Stores secret API keys
├── .env.example          # Template for environment variables
├── .gitignore            # Specifies files for Git to ignore
├── README.md             # This file
├── REPORT.md             # Project report on approach, challenges, etc.
├── requirements.txt      # List of Python dependencies
├── wine_business_info.md # The source document for the winery knowledge base
│
├── agent.py              # Constructs the core LangGraph agent and its logic
├── config.py             # Centralized configuration (model names, paths)
├── knowledge_base.py     # Handles creating and loading the FAISS vector store
├── main.py               # Entry point for the command-line interface
├── tools.py              # Defines all tools available to the agent (weather, search)
└── ui.py                 # Defines and runs the Streamlit web interface
```

## 🔮 Future Improvements

-   **Conversational Memory:** Integrate a memory module to allow for natural follow-up questions and context retention across turns.
-   **Booking & Reservation Tool:** Implement a new tool that can simulate booking a tasting or tour by interacting with a mock API.
-   **Deployment:** Deploy the Streamlit application to a public cloud service like Streamlit Community Cloud or Hugging Face Spaces.
-   **Formal Evaluation:** Set up an evaluation pipeline using a framework like LangSmith to rigorously test the agent's accuracy and performance.

