# tools.py
"""
Defines the collection of tools available to the conversational agent.

This module sets up the agent's capabilities, including:
1.  A retriever tool for searching the winery's knowledge base.
2.  A web search tool for general knowledge queries.
3.  A custom tool for fetching real-time weather information.

Proper error handling and logging are implemented for external API calls.
"""

import os
import logging
import requests
from typing import List

from langchain_core.tools import tool, BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch

from knowledge_base import get_retriever

# Configure basic logging for better debugging of tool errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Weather Tool ---

@tool
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a specified city using the OpenWeatherMap API.

    Args:
        city: The name of the city for which to get the weather (e.g., "Napa, CA").

    Returns:
        A string describing the current weather conditions, or an error message
        if the request fails.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logging.error("OpenWeatherMap API key not found in environment variables.")
        return "Error: Weather service API key is not configured."

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        weather_description = data['weather'][0]['description']
        temp = data['main']['temp']
        
        return f"The current weather in {city} is {temp}°C with {weather_description}."

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while fetching weather for {city}: {http_err}")
        if http_err.response.status_code == 401:
            return "I am sorry, I cannot get the weather. The weather service API key is invalid."
        return f"I am sorry, I cannot fetch the weather for {city}. The service reported an error."

    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request exception occurred while fetching weather: {req_err}")
        return "I am sorry, there was a problem connecting to the weather service."

    except KeyError:
        logging.error(f"KeyError: Could not parse weather data for {city}. City may not be found.")
        return f"I am sorry, I could not find weather information for {city}. Please check the city name."


# --- Tool Aggregation ---

def get_all_tools() -> List[BaseTool]:
    """
    Initializes and aggregates all the tools available to the agent.

    This function serves as a centralized factory for the agent's capabilities.

    Returns:
        A list of BaseTool objects that the agent can use.
    """
    # 1. Winery Knowledge Base Tool
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="winery_information_search",
        description=(
            "Search for information about the Golden Vine Winery. Use this for any "
            "questions about the winery's history, its wines, visiting hours, tours, and location."
        )
    )

    # 2. Web Search Tool
    web_search_tool = TavilySearch()

    return [retriever_tool, web_search_tool, get_weather]