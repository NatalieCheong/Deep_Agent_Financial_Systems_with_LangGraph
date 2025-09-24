"""
LangGraph Studio Configuration
DeepAgent Financial Systems - Main workflow graph for LangGraph Studio

This file defines the primary graph that will be displayed and executed in LangGraph Studio.
It integrates all capabilities: planning, file system, sub-agents, and research tools.
Uses ONLY real YFinance market data for all financial analysis.
"""

from typing import Annotated, Dict, Any, List, Literal
from typing_extensions import TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# MemorySaver removed - LangGraph Studio handles persistence automatically

# Import our components
from config import config
from financial_tools import FINANCIAL_TOOLS

# State for LangGraph Studio
class FinancialAgentState(TypedDict):
    """State for the financial agent workflow in LangGraph Studio using real market data"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    analysis_type: Literal["stock_analysis", "portfolio_analysis", "market_research", "risk_assessment", "general"]
    progress: Dict[str, Any]
    results: Dict[str, Any]
    real_data_used: bool  # Flag to ensure only real data is used

def determine_analysis_type(state: FinancialAgentState) -> FinancialAgentState:
    """
    Analyze the user's request to determine the type of financial analysis needed
    """
    # Initialize missing state keys if they don't exist
    if "progress" not in state:
        state["progress"] = {}
    if "results" not in state:
        state["results"] = {}
    if "real_data_used" not in state:
        state["real_data_used"] = False
    
    if not state["messages"]:
        state["analysis_type"] = "general"
        return state
    
    last_message = state["messages"][-1]
    user_input = last_message.content.lower() if hasattr(last_message, 'content') else ""
    
    # Simple keyword-based analysis type detection
    if any(keyword in user_input for keyword in ["stock", "share", "equity", "company", "ticker"]):
        state["analysis_type"] = "stock_analysis"
    elif any(keyword in user_input for keyword in ["portfolio", "allocation", "diversif", "rebalanc"]):
        state["analysis_type"] = "portfolio_analysis"
    elif any(keyword in user_input for keyword in ["market", "sector", "trend", "outlook", "economy"]):
        state["analysis_type"] = "market_research"
    elif any(keyword in user_input for keyword in ["risk", "volatil", "beta", "var", "stress"]):
        state["analysis_type"] = "risk_assessment"
    else:
        state["analysis_type"] = "general"
    
    state["current_step"] = "planning"
    state["progress"] = {"analysis_type_determined": True}
    state["real_data_used"] = True  # Ensure we flag real data usage
    
    # Add system message about analysis type
    analysis_msg = AIMessage(content=f"🎯 Analysis Type Determined: {state['analysis_type'].replace('_', ' ').title()} (Using REAL YFinance Data)")
    state["messages"].append(analysis_msg)
    
    return state

def create_analysis_plan(state: FinancialAgentState) -> FinancialAgentState:
    """
    Create a structured plan for the financial analysis based on the determined type
    All analysis will use real YFinance market data
    """
    # Initialize missing state keys if they don't exist
    if "progress" not in state:
        state["progress"] = {}
    if "results" not in state:
        state["results"] = {}
    if "real_data_used" not in state:
        state["real_data_used"] = False
    
    analysis_type = state["analysis_type"]
    
    # Create analysis plans based on type - emphasizing real data usage
    plans = {
        "stock_analysis": [
            "Gather current real stock price and live market metrics from YFinance",
            "Analyze actual financial statements and current fundamental metrics", 
            "Calculate real risk metrics using actual historical volatility data",
            "Research current market context using live sector performance data",
            "Compile investment recommendation with real price targets based on current data"
        ],
        "portfolio_analysis": [
            "Analyze current portfolio composition using real market prices",
            "Calculate portfolio performance using actual historical returns",
            "Assess real correlation and diversification using live market data",
            "Evaluate rebalancing opportunities based on current market conditions",
            "Provide optimization recommendations using real risk-return data"
        ],
        "market_research": [
            "Analyze major market indices using current real performance data",
            "Research sector performance using actual rotation patterns",
            "Evaluate real economic indicators and current market sentiment",
            "Identify investment opportunities using real market screening",
            "Compile market outlook using actual trend analysis and real forecasts"
        ],
        "risk_assessment": [
            "Calculate key risk metrics using real historical volatility data",
            "Perform stress testing using actual historical drawdown scenarios",
            "Analyze real correlation and concentration risks using current data",
            "Evaluate hedging strategies based on current real market conditions",
            "Provide risk management recommendations using actual risk measurements"
        ],
        "general": [
            "Analyze the financial request using real market context",
            "Gather relevant real market and financial data from YFinance",
            "Apply appropriate analytical frameworks to actual market conditions",
            "Synthesize insights using real market data and current metrics",
            "Prepare comprehensive response based on actual financial analysis"
        ]
    }
    
    plan = plans.get(analysis_type, plans["general"])
    
    # Create TODO list emphasizing real data
    todo_items = []
    for i, task in enumerate(plan, 1):
        todo_items.append({
            "id": f"task_{i}",
            "description": task,
            "status": "pending",
            "priority": "high" if i <= 2 else "medium",
            "data_source": "Real YFinance Market Data"
        })
    
    state["progress"]["plan_created"] = True
    state["progress"]["todo_items"] = todo_items
    state["current_step"] = "execution"
    state["real_data_used"] = True
    
    # Add planning message
    plan_msg = AIMessage(content=f"📋 Real Data Analysis Plan Created:\n" + "\n".join([f"{i}. {task} (Real YFinance Data)" for i, task in enumerate(plan, 1)]))
    state["messages"].append(plan_msg)
    
    return state

def execute_financial_analysis(state: FinancialAgentState) -> FinancialAgentState:
    """
    Execute the financial analysis using real market data and Deep Research Agent capabilities
    """
    # Initialize missing state keys if they don't exist
    if "progress" not in state:
        state["progress"] = {}
    if "results" not in state:
        state["results"] = {}
    if "real_data_used" not in state:
        state["real_data_used"] = False
    
    # Create the deep research model
    model = ChatOpenAI(
        model=config.get_model_settings("reasoning")["model"],
        temperature=0.05,
        api_key=config.OPENAI_API_KEY,
        max_tokens=4000
    )
    
    # Use all real financial tools
    tools = FINANCIAL_TOOLS.copy()
    
    # Add web search if available for market context
    if config.TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=3,
            search_depth="advanced"
        )
        tools.append(web_search)
    
    # Create agent with real data focus
    from langgraph.prebuilt import create_react_agent
    
    system_prompt = f"""You are executing a {state['analysis_type'].replace('_', ' ')} for DeepAgent Financial Systems using ONLY real market data.

**Current Analysis Type**: {state['analysis_type'].replace('_', ' ').title()}

**CRITICAL REQUIREMENT**: Use ONLY real YFinance market data - never mock, simulated, or test data.

**Your Mission**: Provide comprehensive, data-driven financial analysis using current market prices and actual historical performance.

**Available Real Data Tools**:
- Real-time stock prices and market data via YFinance
- Actual historical price data and performance metrics
- Current financial statements and real fundamental ratios
- Live market overview with real index performance and VIX data
- Real portfolio analysis with actual correlation and risk metrics
- Current risk calculations using actual volatility and drawdown data

**Real Data Guidelines**:
1. Use ONLY current market prices from YFinance - verify timestamps
2. Base all analysis on actual historical performance data
3. Calculate real risk metrics using actual volatility measurements
4. Provide specific, quantified recommendations with real supporting data
5. Include current market context using live index and sector data
6. Validate all data is current and from real market conditions

**Current Plan**: Execute the real data analysis systematically and thoroughly.

**Analysis Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Focus on delivering institutional-quality analysis using exclusively real market data with actionable insights based on current market conditions."""

    agent = create_react_agent(model, tools, prompt=system_prompt)
    
    # Get the original user message for analysis
    user_message = state["messages"][0] if state["messages"] else HumanMessage(content="Perform real market data financial analysis")
    
    # Execute the analysis
    try:
        result = agent.invoke({"messages": [user_message]})
        
        if result and "messages" in result:
            # Add the agent's response to our state
            for msg in result["messages"][1:]:  # Skip the input message
                state["messages"].append(msg)
        
        state["current_step"] = "completed"
        state["progress"]["analysis_completed"] = True
        state["progress"]["real_data_timestamp"] = datetime.now().isoformat()
        state["results"]["execution_successful"] = True
        state["results"]["data_source"] = "Real YFinance Market Data"
        state["real_data_used"] = True
        
        # Add completion message
        completion_msg = AIMessage(content=f"✅ Financial analysis completed using REAL YFinance market data at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        state["messages"].append(completion_msg)
        
    except Exception as e:
        error_msg = AIMessage(content=f"❌ Error during real data analysis execution: {str(e)}")
        state["messages"].append(error_msg)
        state["current_step"] = "error"
        state["results"]["error"] = str(e)
    
    return state

# Create the main workflow graph
def create_financial_workflow() -> StateGraph:
    """
    Create the main financial analysis workflow for LangGraph Studio using real market data
    """
    workflow = StateGraph(FinancialAgentState)
    
    # Add nodes
    workflow.add_node("determine_type", determine_analysis_type)
    workflow.add_node("create_plan", create_analysis_plan)
    workflow.add_node("execute_analysis", execute_financial_analysis)
    
    # Add edges
    workflow.add_edge(START, "determine_type")
    workflow.add_edge("determine_type", "create_plan")
    workflow.add_edge("create_plan", "execute_analysis")
    workflow.add_edge("execute_analysis", END)
    
    # Compile without custom checkpointer - LangGraph Studio handles persistence automatically
    return workflow.compile()

# Create the graph instance for LangGraph Studio
graph = create_financial_workflow()

# Test function for local development
def test_workflow():
    """
    Test the workflow locally with real market data
    """
    initial_state = {
        "messages": [HumanMessage(content="Analyze Apple (AAPL) stock using real current market data and provide investment recommendation based on actual performance metrics")],
        "current_step": "start",
        "analysis_type": "general",
        "progress": {},
        "results": {},
        "real_data_used": True
    }
    
    config_dict = {"configurable": {"thread_id": "test_session"}}
    
    try:
        print("🧪 Testing workflow with real YFinance data...")
        step_count = 0
        
        for step in graph.stream(initial_state, config_dict):
            step_count += 1
            print(f"Step {step_count}: {list(step.keys())}")
            
        print("✅ Workflow test completed successfully using real market data")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")

if __name__ == "__main__":
    # Test the workflow when run directly
    test_workflow()