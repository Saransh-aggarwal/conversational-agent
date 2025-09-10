# tools.py
import os
import requests
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from knowledge_base import get_retriever

def get_weather(city: str):
    """
    Fetches the current weather for a given city using the OpenWeatherMap API.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OpenWeatherMap API key not found."
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"} # Use metric for Celsius
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        weather_description = data['weather'][0]['description']
        temp = data['main']['temp']
        
        return f"The current weather in {city} is {temp}°C with {weather_description}."
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except KeyError:
        return f"Error: Could not parse weather data for {city}. City may not be found."


def get_all_tools():
    """
    Initializes and returns a list of all tools available to the agent.
    """
    # 1. Winery Knowledge Base Tool
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "winery_information_search",
        "Search for information about the Golden Vine Winery. Use this for any questions about the winery's history, its wines, visiting hours, tours, and location."
    )

    # 2. Web Search Tool
    web_search_tool = TavilySearchResults()

    # We are not creating a @tool for the weather function yet, as we will use a different method 
    # in the graph itself for simplicity. You can wrap it in a @tool decorator for more complex scenarios.
    # We will simply map the function to a tool call in the agent.

    return [retriever_tool, web_search_tool]