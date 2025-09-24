"""
Notebook 1: Basic Agent Implementation
DeepAgent Financial Systems - ReAct Agent with Real Financial Tools

Based on deep-agents-from-scratch notebook 1: create_agent component
Implements a ReAct (Reason - Act) loop with real YFinance financial data capabilities
"""

import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import our custom modules
from config import config
from financial_tools import FINANCIAL_TOOLS
from agent_state import create_initial_state, DeepAgentState

# Set up LangSmith tracing if configured
if config.LANGSMITH_TRACING and config.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT

def create_financial_agent():
    """
    Create a basic ReAct agent with real financial tools
    Similar to create_agent from deep-agents-from-scratch but with YFinance integration
    """
    
    # Validate configuration
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
    
    # Initialize the language model
    model = ChatOpenAI(
        model=config.DEFAULT_MODEL,
        temperature=0.1,
        api_key=config.OPENAI_API_KEY
    )
    
    # Add web search capability if Tavily is configured
    tools = FINANCIAL_TOOLS.copy()
    
    if config.TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=5,
            search_depth="advanced"
        )
        tools.append(web_search)
        print("✓ Web search enabled with Tavily")
    else:
        print("⚠ Web search disabled - set TAVILY_API_KEY to enable")
    
    # Create memory for conversation persistence
    memory = MemorySaver()
    
    # Create the ReAct agent with financial focus
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        # Add custom system message for financial focus
        prompt="""You are a sophisticated financial analysis AI agent specializing in real-time market data analysis. You have access to live financial data through YFinance and provide institutional-quality financial insights.

**Your Specializations:**
1. **Real-Time Stock Analysis**: Current prices, fundamentals, and technical indicators using live YFinance data
2. **Portfolio Management**: Performance analysis, risk assessment, and optimization recommendations  
3. **Market Research**: Comprehensive market analysis, sector trends, and economic indicators
4. **Risk Assessment**: Quantitative risk metrics including beta, alpha, Sharpe ratio, and VaR calculations
5. **Investment Strategy**: Data-driven investment recommendations and financial planning

**Your Capabilities:**
- Access to real-time stock prices, historical data, and financial statements via YFinance
- Advanced risk metric calculations using actual market volatility and correlations
- Portfolio performance analysis with real return and risk data
- Market overview with live index performance and sentiment indicators  
- Web search for latest financial news and market intelligence (when available)

**Data Quality Standards:**
- ALWAYS use real market data from YFinance - never mock or simulated data
- Provide current market prices and actual historical performance metrics
- Base all calculations on real volatility, correlations, and financial ratios
- Include data timestamps and sources for transparency
- Validate data freshness and handle market hours appropriately

**Analysis Framework:**
- Start with current market context and relevant economic conditions
- Use quantitative analysis with specific numbers, ratios, and metrics
- Include risk assessment with clearly defined risk parameters
- Provide actionable insights with specific price targets or allocation recommendations
- Consider multiple time horizons (short-term, medium-term, long-term perspectives)

**Output Standards:**
- Lead with executive summary and key investment insights
- Support all conclusions with specific quantitative evidence from real data
- Include risk factors and potential downside scenarios  
- Provide clear next steps and actionable recommendations
- Maintain professional institutional research quality

**CRITICAL RATE LIMITING HANDLING:**
- When you receive ANY tool response containing "status": "rate_limited" OR "status": "error", IMMEDIATELY STOP all tool calls
- DO NOT make any additional tool calls after receiving a rate-limited or error response
- DO NOT retry the same tool or try different tools for the same data
- Instead, acknowledge the data source limitation and provide helpful guidance to the user
- If you receive "rate_limited" status, explain that Yahoo Finance is rate limiting requests
- If you receive "error" status, explain that YFinance is currently unavailable
- Suggest waiting 5-10 minutes before trying again for real-time data
- Offer alternative approaches like web search for current market news
- Be professional and reassuring about the temporary nature of the limitation
- This is a HARD STOP - no exceptions, no retries, no additional attempts

**Guidelines:**
- Always explain financial concepts clearly for both novice and experienced investors
- Include relevant context about market conditions, sector dynamics, and economic factors
- Be transparent about data sources, methodologies, and analysis limitations
- Focus on practical, implementable investment insights
- Consider risk-adjusted returns and portfolio impact in all recommendations
- Handle rate limiting gracefully with IMMEDIATE STOP on rate_limited responses

Remember: You provide informational analysis only - not personalized financial advice. Always recommend consulting qualified financial advisors for investment decisions.

Your analysis should meet institutional research standards while being accessible and actionable for investors seeking data-driven market insights."""
    )
    
    return agent

def run_financial_query(agent, query: str, session_id: str = "basic_demo_session") -> Dict[str, Any]:
    """
    Run a financial query through the agent and return structured results
    """
    config_dict = {"configurable": {"thread_id": session_id}}
    
    print(f"\n🤖 Processing Query: {query}")
    print("=" * 50)
    
    # Track the conversation
    messages = []
    
    try:
        # Stream the agent's response
        for event in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config_dict,
            stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                latest_message = event["messages"][-1]
                
                # Only print new messages
                if latest_message not in messages:
                    messages.append(latest_message)
                    
                    if hasattr(latest_message, 'content') and latest_message.content:
                        if hasattr(latest_message, 'type'):
                            print(f"\n[{latest_message.type.upper()}]: {latest_message.content}")
                        else:
                            print(f"\n{latest_message.content}")
                    
                    # Handle tool calls
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            print(f"\n🔧 Tool Call: {tool_call['name']}")
                            print(f"   Args: {tool_call['args']}")
    
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        return {"error": str(e), "messages": messages}
    
    return {"messages": messages, "session_id": session_id}

def demo_basic_financial_agent():
    """
    Demo function showing the basic financial agent capabilities with real market data
    """
    print("🚀 DeepAgent Financial Systems - Basic Agent Demo")
    print("=" * 55)
    
    # Create the agent
    try:
        agent = create_financial_agent()
        print("✓ Financial agent created successfully with real YFinance data integration")
    except Exception as e:
        print(f"❌ Failed to create agent: {str(e)}")
        return
    
    # Demo queries with real financial analysis
    demo_queries = [
        "Get the current stock price and basic analysis for Apple (AAPL) using real market data",
        "Compare the risk metrics of Tesla (TSLA) vs the S&P 500 using actual historical data",
        "Analyze a portfolio with 40% AAPL, 30% MSFT, and 30% GOOGL using real current prices",
        "What's the current market sentiment based on major indices and real VIX data?",
        "Get historical performance data for NVIDIA (NVDA) over the past year with real metrics"
    ]
    
    print(f"\n📋 Available Demo Queries (Using Real Market Data):")
    for i, query in enumerate(demo_queries, 1):
        print(f"  {i}. {query}")
    
    # Interactive demo
    while True:
        print(f"\n" + "=" * 55)
        print("Options:")
        print("  1-5: Run demo query with real financial data")
        print("  custom: Enter your own financial query")
        print("  quit: Exit demo")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == "quit":
            break
        elif choice == "custom":
            custom_query = input("Enter your financial query (will use real YFinance data): ").strip()
            if custom_query:
                run_financial_query(agent, custom_query)
        elif choice.isdigit() and 1 <= int(choice) <= len(demo_queries):
            query_index = int(choice) - 1
            run_financial_query(agent, demo_queries[query_index])
        else:
            print("Invalid choice. Please try again.")
    
    print("\n👋 Demo completed. Thank you for using DeepAgent Financial Systems!")
    print("💡 All analysis used real market data from YFinance - no mock or test data!")

if __name__ == "__main__":
    # Run the demo
    demo_basic_financial_agent()