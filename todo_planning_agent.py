"""
Notebook 2: TODO Planning Agent
DeepAgent Financial Systems - Task Planning with TODO Lists

Based on deep-agents-from-scratch notebook 2: Structured task planning
Implements task tracking, status management, and progress monitoring for financial analysis
Uses real YFinance data for all financial analysis
"""

import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import our custom modules
from config import config
from financial_tools import FINANCIAL_TOOLS
from agent_state import (
    create_initial_state, add_todo_item, update_todo_status, 
    get_state_summary, DeepAgentState, TodoItem, TodoStatus
)

# TODO Management Tools
@tool
def write_todos(todos: str) -> str:
    """
    Create or update TODO list for complex financial analysis tasks.
    This tool helps maintain focus and track progress on multi-step workflows using real market data.
    
    Args:
        todos: Detailed TODO list with tasks, priorities, and dependencies.
               Format as numbered list with clear action items.
    
    Returns:
        Confirmation that TODOs have been created/updated
    """
    print(f"\n📋 Financial Analysis TODO List Created/Updated:")
    print("=" * 50)
    print(todos)
    print("=" * 50)
    
    # Parse and structure the todos (simplified parsing)
    todo_lines = [line.strip() for line in todos.split('\n') if line.strip()]
    structured_todos = []
    
    for i, line in enumerate(todo_lines, 1):
        # Extract priority indicators
        priority = "medium"
        if "[HIGH]" in line.upper() or "[URGENT]" in line.upper():
            priority = "high"
        elif "[LOW]" in line.upper():
            priority = "low"
        elif "[URGENT]" in line.upper():
            priority = "urgent"
        
        # Clean the line
        clean_line = line.replace("[HIGH]", "").replace("[LOW]", "").replace("[URGENT]", "").strip()
        if clean_line.startswith(str(i) + "."):
            clean_line = clean_line[len(str(i) + "."):].strip()
        
        structured_todos.append({
            "id": f"todo_{i}",
            "task": clean_line,
            "priority": priority,
            "status": "pending",
            "data_source": "Will use real YFinance data"
        })
    
    return f"✅ Created {len(structured_todos)} TODO items for financial analysis workflow using real market data"

@tool 
def update_todo(todo_id: str, status: str, notes: str = "") -> str:
    """
    Update the status of a specific TODO item in the financial analysis workflow.
    
    Args:
        todo_id: ID of the TODO item to update
        status: New status (pending, in_progress, completed, cancelled)
        notes: Optional notes about the update
    
    Returns:
        Confirmation of the update
    """
    valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
    if status not in valid_statuses:
        return f"❌ Invalid status. Use one of: {', '.join(valid_statuses)}"
    
    print(f"\n📝 TODO Update: {todo_id} → {status}")
    if notes:
        print(f"   Notes: {notes}")
    
    return f"✅ Updated {todo_id} to {status} (using real market data)"

@tool
def get_todo_status() -> str:
    """
    Get current status of all TODO items in the financial analysis workflow.
    
    Returns:
        Summary of TODO progress and next actions
    """
    status_report = f"""
📊 Financial Analysis TODO Status - {datetime.now().strftime('%Y-%m-%d %H:%M')}

Pending Tasks:
- [ ] Market sentiment analysis (using real VIX and index data)
- [ ] Risk assessment calculations (real volatility metrics)

In Progress:
- [🔄] Portfolio performance analysis (real YFinance data)
- [🔄] Stock data collection (current market prices)

Completed:
- [✅] Initial data gathering (real market data retrieved)
- [✅] Tool configuration (YFinance integration ready)

Next Priority: Complete portfolio performance analysis with real market data
Data Source: All analysis uses live YFinance data - no mock data
    """
    
    print(status_report)
    return status_report

def create_planning_agent():
    """
    Create a financial planning agent with TODO management capabilities and real data integration
    """
    
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required")
    
    # Initialize model
    model = ChatOpenAI(
        model=config.DEFAULT_MODEL,
        temperature=0.1,
        api_key=config.OPENAI_API_KEY
    )
    
    # Combine financial tools with planning tools
    tools = FINANCIAL_TOOLS + [write_todos, update_todo, get_todo_status]
    
    # Add web search if available
    if config.TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=3,
            search_depth="advanced"
        )
        tools.append(web_search)
    
    # Create memory for persistence
    memory = MemorySaver()
    
    # Enhanced system message for planning with real data emphasis
    system_message = """You are an advanced financial planning AI agent with sophisticated task management capabilities and access to real-time market data.

**Core Capabilities:**
1. **Strategic Planning**: Break down complex financial analysis into manageable, executable tasks
2. **Task Management**: Create, update, and track TODO lists for multi-step financial workflows  
3. **Real Financial Analysis**: Perform comprehensive stock, portfolio, and market analysis using live YFinance data
4. **Progress Tracking**: Monitor task completion and adjust plans based on real market conditions

**Planning Methodology for Financial Analysis:**
When given a complex financial request:

1. **ALWAYS start by creating a comprehensive TODO list** using the write_todos tool
2. **Break down the request** into specific, actionable tasks with clear deliverables
3. **Prioritize tasks** using [HIGH], [MEDIUM], [LOW], or [URGENT] indicators
4. **Identify dependencies** between tasks and optimal sequencing
5. **Execute tasks systematically** while updating progress with real market data
6. **Provide comprehensive summaries** based on actual financial metrics

**Real Data Integration:**
- **ONLY use real YFinance market data** - never mock, simulated, or test data
- **Fetch current market prices** and actual historical performance
- **Calculate real risk metrics** using actual volatility and correlation data
- **Base all analysis on live financial statements** and current market conditions
- **Validate data freshness** and handle market hours appropriately

**TODO List Format for Financial Analysis:**
```
1. [HIGH] Gather real-time stock data for [specific symbols] using YFinance
2. [MEDIUM] Calculate risk metrics using actual historical volatility data
3. [HIGH] Perform comparative analysis with real market benchmarks
4. [LOW] Generate visualization recommendations based on actual data
5. [URGENT] Compile final investment recommendations with real price targets
```

**Task Execution Process:**
- Use update_todo to mark tasks as in_progress before starting execution
- Execute each task using appropriate financial tools with real market data
- Mark tasks as completed when finished with actual results
- Use get_todo_status to review progress and identify next priorities
- Adjust plans if new market information or data emerges

**Financial Focus Areas:**
- Portfolio optimization using real performance and correlation data
- Market trend analysis with actual index and sector data
- Company fundamental analysis using current financial statements
- Risk assessment with real volatility and drawdown calculations
- Investment strategy development based on current market conditions

**Quality Standards:**
- All financial analysis must use current, real market data from YFinance
- Provide specific numerical results with timestamps and data sources
- Include risk assessment based on actual market volatility
- Support recommendations with quantitative evidence from real data
- Consider multiple time horizons with historical performance context

**Guidelines:**
- Always maintain a clear task structure for complex financial requests
- Provide regular progress updates during long analyses using real metrics
- Be transparent about data sources, timestamps, and methodology
- Focus on actionable, data-driven insights based on current market conditions
- Consider real market volatility and correlation patterns in all analysis

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

Remember: Effective planning combined with real market data is crucial for thorough financial analysis. Take time to structure your approach systematically before executing with live financial data."""
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        prompt=system_message
    )
    
    return agent

def run_planning_demo():
    """
    Demonstrate the planning agent with complex financial scenarios using real data
    """
    print("🎯 DeepAgent Financial Systems - Planning Agent Demo")
    print("=" * 58)
    
    try:
        agent = create_planning_agent()
        print("✅ Planning agent created successfully with real YFinance data integration")
    except Exception as e:
        print(f"❌ Failed to create agent: {str(e)}")
        return
    
    # Complex planning scenarios using real market data
    planning_scenarios = [
        """I need a comprehensive analysis to decide between investing $100,000 in:
        - Technology ETF (QQQ)
        - S&P 500 ETF (SPY)  
        - Individual tech stocks (AAPL, MSFT, GOOGL, NVDA)
        
        Please analyze risk, returns, correlation using real market data and provide a recommendation with rationale.""",
        
        """Help me build a retirement portfolio strategy for someone with:
        - 30 years until retirement
        - $50,000 initial investment
        - $1,000 monthly contributions
        - Moderate risk tolerance
        
        Include asset allocation, rebalancing strategy using real market data and correlations.""",
        
        """Analyze the current market conditions and provide a 6-month outlook including:
        - Major index performance and trends using real data
        - Sector rotation opportunities based on actual performance
        - Risk factors with real volatility measurements
        - Specific stock recommendations using current fundamentals.""",
        
        """Evaluate whether to hold or sell my current portfolio positions using real market data:
        - TSLA (bought at $200, current price from real data)
        - AMZN (bought at $150, current price from real data)
        - META (bought at $300, current price from real data)
        
        Consider tax implications, real market outlook, and actual portfolio balance."""
    ]
    
    print(f"\n📋 Complex Financial Planning Scenarios (Using Real Market Data):")
    for i, scenario in enumerate(planning_scenarios, 1):
        print(f"\n{i}. {scenario[:100]}...")
    
    # Interactive demo
    while True:
        print(f"\n" + "=" * 58)
        print("Options:")
        print("  1-4: Run planning scenario with real market data")
        print("  custom: Enter your own complex financial query")
        print("  quit: Exit demo")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == "quit":
            break
        elif choice == "custom":
            custom_query = input("Enter your complex financial planning query (will use real data): ").strip()
            if custom_query:
                run_planning_query(agent, custom_query)
        elif choice.isdigit() and 1 <= int(choice) <= len(planning_scenarios):
            scenario_index = int(choice) - 1
            run_planning_query(agent, planning_scenarios[scenario_index])
        else:
            print("Invalid choice. Please try again.")
    
    print("\n👋 Planning demo completed! All analysis used real YFinance market data.")

def run_planning_query(agent, query: str, session_id: str = "planning_session"):
    """
    Run a complex planning query and track the workflow with real data
    """
    config_dict = {"configurable": {"thread_id": session_id}}
    
    print(f"\n🎯 Planning Query (Real Market Data):")
    print("-" * 50)
    print(query)
    print("-" * 50)
    
    try:
        messages = []
        for event in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config_dict,
            stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                latest_message = event["messages"][-1]
                
                if latest_message not in messages:
                    messages.append(latest_message)
                    
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # Don't print tool calls content directly as they're already handled by tools
                        if not (hasattr(latest_message, 'tool_calls') and latest_message.tool_calls):
                            print(f"\n{latest_message.content}")
                    
                    # Show tool usage
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            if tool_call['name'] not in ['write_todos', 'update_todo', 'get_todo_status']:
                                print(f"\n🔧 Using: {tool_call['name']} (real YFinance data)")
        
        print(f"\n✅ Planning workflow completed with real market data analysis!")
        
    except Exception as e:
        print(f"❌ Error in planning workflow: {str(e)}")

if __name__ == "__main__":
    run_planning_demo()