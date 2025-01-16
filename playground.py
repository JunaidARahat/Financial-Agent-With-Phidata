import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables from .env file
load_dotenv()
PHI_API_KEY = os.getenv("PHI_API_KEY")

if not PHI_API_KEY:
    raise ValueError("PHI_API_KEY not found. Please check your .env file.")

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,  # Ensure consistent key usage
    markdown=True,
)

# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,  # Ensure consistent key usage
    markdown=True,
)

# Create the Playground app
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

# Serve the Playground app
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
