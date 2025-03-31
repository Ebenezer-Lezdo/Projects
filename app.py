import os
import json
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Annotated, List
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.coding import LocalCommandLineCodeExecutor
from tavily import TavilyClient
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf
from tavily import TavilyClient
import yfinance as yf
from autogen import ConversableAgent
import streamlit as st

# Set up the Streamlit app
st.set_page_config(page_title="Agentic AI Assistant", layout="wide")

# Sidebar for API keys and settings
with st.sidebar:
    st.title("Configuration")
    groq_api_key = st.text_input("GROQ API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    openweather_api_key = st.text_input("OpenWeather API Key", type="password")
    
    st.markdown("---")
    st.markdown("### Tools Status")
    groq_status = st.empty()
    tavily_status = st.empty()
    weather_status = st.empty()
   
    
    if groq_api_key:
        groq_status.success("Groq API: Configured")
    else:
        groq_status.warning("Groq API: Not Configured")
    
    if tavily_api_key:
        tavily_status.success("Tavily API: Configured")
    else:
        tavily_status.warning("Tavily API: Not Configured")
        
    if openweather_api_key:
        weather_status.success("OpenWeather API: Configured")
    else:
        weather_status.warning("OpenWeather API: Not Configured")

# Main app area
st.title("Agentic AI Assistant")
st.markdown("""
This assistant can help with:
- Web searches
- Weather forecasts
- Data visualization
- Stock market analysis
- Code generation and debugging
- Research Assistant
""")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                st.image(img_path)

# User input
user_input = st.chat_input("What would you like help with today?")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Set up the environment with the provided API keys
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["OPENWEATHER_API_KEY"] = openweather_api_key
    tavily_client = TavilyClient(api_key=tavily_api_key)

    # Configure Groq API
    config_list = [{
        "model": "llama-3.3-70b-versatile",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "api_type": "groq"
    }]

    # Create a directory to store code files from the code executor
    work_dir = Path("coding")
    work_dir.mkdir(exist_ok=True)
    code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    

    from pathlib import Path


    # Display folder structure recursively
    def display_folder_structure(path, indent=0):
        if path.is_dir():
            # st.write("  " * indent + f"📁 {path.name}/")
            for item in sorted(path.iterdir()):
                display_folder_structure(item, indent + 1)
        else:
            st.write("  " * indent + f"📄 {path.name}")

    st.subheader("Download Visualized Charts here")
    if work_dir.exists():
        display_folder_structure(work_dir)
    else:
        st.warning("Work directory does not exist!")
        
    import base64

    # Display files with download buttons
    for file in work_dir.glob("*"):
        if file.is_file():
            with open(file, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"Download {file.name}",
                data=data,
                file_name=file.name,
                mime="text/plain" if file.suffix == ".txt" else None
            )

    # Initialize the assistant
    assistant = AssistantAgent(
        name="groq_assistant",
        system_message="""
        You are a highly capable Agentic Generative AI with access to the following tools: `weather_forecast`, `generate_bar_chart`, `tavily_search`, and `visualize_stock_data`. Your goal is to accurately and helpfully answer the user's query by leveraging these tools effectively.

        Here's how you should approach each task:

        1. **Understand the Request:**  Carefully analyze the user's query to determine the information they need and the desired output format. Always run code, debug it and give.

        2. **Plan Your Approach:**  Break down the task into smaller, logical steps.  Select the most appropriate tools for each step, avoiding unnecessary tool use. Consider the tool descriptions to choose the best option.

        3. **Execute and Observe:**  Execute your plan, calling the necessary tools and carefully recording the results.  Here are examples of how to call each tool:
            * `weather_forecast(location="London", unit="celsius")`
            * `tavily_search(query="latest AI advancements")`
            * `generate_bar_chart(categories=["A", "B", "C"], values=[10, 20, 30], title="Example Chart")`
            * `visualize_stock_data(ticker="AAPL", start_date="2024-01-01")`

        4. **Synthesize and Respond:**  Combine the results from each step to provide a clear, concise, and helpful answer to the user's query. Briefly summarize the steps you took and the tools you used.

        If a tool returns an error or unexpected result, try to recover gracefully.  Consider alternative approaches or tools.  If you cannot fulfill the request, explain why.

        Remember to prioritize providing a user-friendly and informative response.

        Once you believe the task is complete, say **TERMINATE**.
        """,
        llm_config={"config_list": config_list},
    )

    # Create a user proxy agent with code execution capabilities
    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"executor": code_executor},
        is_termination_msg=lambda msg: msg.get("content") and ("TERMINATE" in msg["content"] or "task completed" in msg["content"].lower()),
        human_input_mode="NEVER"
    )

    # Function definitions (same as before)
    def tavily_search(query: str) -> str:
        """Fetches search results using Tavily API."""
        try:
            response = tavily_client.search(query)
            return json.dumps(response, indent=2)
        except Exception as e:
            return f"Error fetching search results: {str(e)}"

    def fetch_stock_data(ticker: str, start_date: str = "2024-01-01") -> str:
        """
        Fetches historical stock data for a given ticker symbol starting from the specified date.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date)
            if data.empty:
                return f"No data found for ticker symbol '{ticker}' starting from {start_date}."
            # Save the data to a CSV file
            file_path = work_dir / f"{ticker}_stock_data.csv"
            data.to_csv(file_path)
            return f"Stock data for {ticker} saved to {file_path}."
        except Exception as e:
            return f"An error occurred while fetching stock data: {str(e)}"

    def get_current_weather(location: str, unit: str) -> str:
        """Fetches weather data from OpenWeatherMap API."""
        API_KEY = os.environ.get("OPENWEATHER_API_KEY")
        BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

        params = {
            "q": location,
            "appid": API_KEY,
            "units": "metric" if unit == "celsius" else "imperial"
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "location": data["name"],
                "temperature": data["main"]["temp"],
                "unit": unit
            })
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    def weather_forecast(
        location: Annotated[str, "City name"],
        unit: Annotated[str, "Temperature unit (fahrenheit/celsius)"] = "fahrenheit"
    ) -> str:
        """Fetches weather data from an external source."""
        weather_details = get_current_weather(location=location, unit=unit)
        weather = json.loads(weather_details)

        if weather["temperature"] == "unknown":
            return f"Could not retrieve weather for {weather['location']}."

        return f"{weather['location']} will be {weather['temperature']} degrees {weather['unit']}."

    def generate_bar_chart(categories: List[str], values: List[int], title: str = "Bar Chart") -> str:
        """Generates a bar chart, saves it as an image, and displays it."""
        plt.figure(figsize=(8, 6))
        plt.bar(categories, values, color='skyblue')
        plt.xlabel("Categories")
        plt.ylabel("Values")
        plt.title(title)
        plt.xticks(rotation=45)

        # Save the plot and display it
        file_path = work_dir / "bar_chart.png"
        plt.savefig(file_path)
        plt.close()

        return f"Bar chart saved as {file_path}."

    def visualize_stock_data(ticker: str, start_date: str = "2024-01-01") -> str:
        """
        Generates advanced visualizations for a given stock ticker, including a line plot of closing prices
        and a candlestick chart, starting from the specified date.
        """
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date)
            if data.empty:
                return f"No data found for ticker symbol '{ticker}' starting from {start_date}."

            # Set the plot style
            sns.set(style="whitegrid")

            # Create a line plot of the closing prices
            plt.figure(figsize=(14, 7))
            sns.lineplot(x=data.index, y=data['Close'], label='Closing Price', color='b')
            plt.title(f"{ticker} Closing Prices Since {start_date}")
            plt.xlabel("Date")
            plt.ylabel("Closing Price (USD)")
            plt.legend()
            plt.xticks(rotation=45)
            line_plot_path = work_dir / f"{ticker}_closing_prices.png"
            plt.savefig(line_plot_path)
            plt.close()

            # Create a candlestick chart
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc)
            candlestick_plot_path = work_dir / f"{ticker}_candlestick_chart.png"
            mpf.plot(data, type='candle', style=s, title=f"{ticker} Candlestick Chart", ylabel='Price (USD)',
                     savefig=dict(fname=candlestick_plot_path, dpi=100, bbox_inches='tight'))

            return f"Visualizations saved as {line_plot_path} and {candlestick_plot_path}."
        except Exception as e:
            return f"An error occurred while generating visualizations: {str(e)}"

    # Register all functions
    register_function(
        weather_forecast,
        caller=assistant,
        executor=user_proxy,
        name="weather_forecast",
        description="Fetch the weather forecast for a city in Fahrenheit or Celsius."
    )

    register_function(
        generate_bar_chart,
        caller=assistant,
        executor=user_proxy,
        name="generate_bar_chart",
        description="Generate a bar chart from given categories and values."
    )

    register_function(
        tavily_search,
        caller=assistant,
        executor=user_proxy,
        name="tavily_search",
        description="Search the web using Tavily API."
    )

    register_function(
        visualize_stock_data,
        caller=assistant,
        executor=user_proxy,
        name="visualize_stock_data",
        description="Generate advanced visualizations for a given stock ticker, including line and candlestick charts."
    )

    # Start the conversation
    with st.spinner("Processing your request..."):
        user_proxy.initiate_chat(assistant, message=user_input)
        
        # Get the last message from the assistant
        last_message = assistant.last_message()
        if last_message:
            # Check for generated images
            images = []
            if "bar_chart.png" in last_message["content"]:
                images.append(work_dir / "bar_chart.png")
            if "_closing_prices.png" in last_message["content"]:
                ticker = user_input.split("ticker=")[1].split(" ")[0].strip('"')
                images.append(work_dir / f"{ticker}_closing_prices.png")
                images.append(work_dir / f"{ticker}_candlestick_chart.png")
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": last_message["content"],
                "images": images
            })
            
            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(last_message["content"])
                for img_path in images:
                    if img_path.exists():
                        st.image(str(img_path))