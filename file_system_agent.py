"""
Notebook 3: File System Agent
DeepAgent Financial Systems - Virtual File System for Context Management

Based on deep-agents-from-scratch notebook 3: Virtual File System
Implements context offloading through file operations with real YFinance data
Enables agent "memory" across conversation turns and reduces token usage
"""

import os
import json
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
from agent_state import create_initial_state, write_file, read_file, list_files, DeepAgentState

# Virtual File System Tools
@tool
def ls(directory: str = "/") -> str:
    """
    List files and directories in the virtual file system.
    Used for exploring saved financial data, analysis results, and reports.
    
    Args:
        directory: Directory path to list (default: root "/")
    
    Returns:
        List of files and directories with metadata
    """
    # This would access the actual state in a real implementation
    # For demo, we'll simulate a file system with financial data
    
    sample_files = {
        "/": [
            "portfolio_analysis.json",
            "market_research/",
            "stock_data/", 
            "risk_reports/",
            "todo_list.md",
            "investment_strategy.md"
        ],
        "/stock_data": [
            "AAPL_analysis.json",
            "TSLA_metrics.json",
            "MSFT_financials.json",
            "tech_sector_comparison.csv"
        ],
        "/market_research": [
            "sp500_trends.json",
            "sector_rotation_analysis.md",
            "economic_indicators.json",
            "volatility_report.md"
        ],
        "/risk_reports": [
            "portfolio_risk_metrics.json",
            "correlation_analysis.json",
            "stress_test_results.md"
        ]
    }
    
    files = sample_files.get(directory, [])
    
    if not files:
        return f"Directory '{directory}' is empty or does not exist."
    
    result = f"📁 Contents of '{directory}':\n"
    for file in files:
        if file.endswith("/"):
            result += f"  📁 {file}\n"
        else:
            result += f"  📄 {file}\n"
    
    return result

@tool
def read_file(filename: str) -> str:
    """
    Read content from a file in the virtual file system.
    Used to retrieve saved financial analysis, market data, or research notes.
    
    Args:
        filename: Path to the file to read
    
    Returns:
        File content or error message
    """
    # In a real implementation, this would read from the agent state
    # For demo, we'll return sample financial data based on filename
    
    if "AAPL" in filename.upper():
        return json.dumps({
            "symbol": "AAPL",
            "analysis_date": datetime.now().isoformat(),
            "current_price": 185.25,
            "analysis": {
                "recommendation": "BUY",
                "target_price": 205.00,
                "risk_level": "Medium",
                "key_metrics": {
                    "pe_ratio": 28.5,
                    "revenue_growth": "8.2%",
                    "profit_margin": "23.1%"
                }
            },
            "notes": "Strong fundamentals with consistent iPhone revenue and growing services segment"
        }, indent=2)
    
    elif "portfolio" in filename.lower():
        return json.dumps({
            "portfolio_name": "Tech Growth Portfolio",
            "last_updated": datetime.now().isoformat(),
            "positions": [
                {"symbol": "AAPL", "weight": 0.35, "value": 35000},
                {"symbol": "MSFT", "weight": 0.25, "value": 25000},
                {"symbol": "GOOGL", "weight": 0.20, "value": 20000},
                {"symbol": "NVDA", "weight": 0.20, "value": 20000}
            ],
            "total_value": 100000,
            "ytd_return": 12.5,
            "risk_metrics": {
                "volatility": 18.2,
                "sharpe_ratio": 1.35,
                "max_drawdown": -8.5
            }
        }, indent=2)
    
    elif "market_research" in filename.lower() or "trends" in filename.lower():
        return """# Market Trends Analysis
Date: {date}

## Key Findings:
- S&P 500 showing resilience despite economic uncertainty
- Technology sector leading with 15% YTD gains
- Energy sector underperforming due to oil price volatility
- Interest rate environment creating headwinds for growth stocks

## Sector Performance (YTD):
- Technology: +15.2%
- Healthcare: +8.7%
- Financials: +6.1%
- Energy: -3.8%
- Utilities: +2.1%

## Market Outlook:
Cautiously optimistic for Q4 with focus on earnings quality and Fed policy signals.
""".format(date=datetime.now().strftime("%Y-%m-%d"))
    
    else:
        return f"❌ File '{filename}' not found in virtual file system."

@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the virtual file system.
    Used to save financial analysis results, market research, or notes for later reference.
    
    Args:
        filename: Path where to save the file
        content: Content to write to the file
    
    Returns:
        Confirmation message
    """
    # Validate filename
    if not filename or filename.startswith("/"):
        filename = filename[1:] if filename.startswith("/") else filename
    
    # Simulate writing to virtual file system
    timestamp = datetime.now().isoformat()
    size = len(content)
    
    print(f"📝 Writing to file: {filename}")
    print(f"   Size: {size} characters")
    print(f"   Timestamp: {timestamp}")
    
    # Show preview of content for financial data
    if size > 200:
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"   Preview: {preview}")
    else:
        print(f"   Content: {content}")
    
    return f"✅ Successfully wrote {size} characters to '{filename}' at {timestamp}"

@tool
def edit_file(filename: str, search_text: str, replace_text: str) -> str:
    """
    Edit an existing file by replacing specific text.
    Useful for updating financial analysis, adding new data, or correcting information.
    
    Args:
        filename: Path to the file to edit
        search_text: Text to search for
        replace_text: Text to replace it with
    
    Returns:
        Confirmation of the edit
    """
    # In a real implementation, this would modify the actual file content
    print(f"✏️ Editing file: {filename}")
    print(f"   Replacing: '{search_text}'")
    print(f"   With: '{replace_text}'")
    
    return f"✅ Successfully updated '{filename}' - replaced '{search_text}' with '{replace_text}'"

def create_file_system_agent():
    """
    Create a financial agent with virtual file system capabilities for context management
    """
    
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required")
    
    # Initialize model
    model = ChatOpenAI(
        model=config.DEFAULT_MODEL,
        temperature=0.1,
        api_key=config.OPENAI_API_KEY
    )
    
    # Combine all tools: financial + file system + planning
    from todo_planning_agent import write_todos, update_todo, get_todo_status
    
    tools = FINANCIAL_TOOLS + [
        ls, read_file, write_file, edit_file,
        write_todos, update_todo, get_todo_status
    ]
    
    # Add web search if available
    if config.TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=3,
            search_depth="advanced"
        )
        tools.append(web_search)
    
    # Create memory
    memory = MemorySaver()
    
    # Enhanced system message for file system integration
    system_message = """You are an advanced financial analysis AI with sophisticated file system capabilities for context management and persistent memory.

**Core Capabilities:**
1. **Financial Analysis**: Real-time stock data analysis using YFinance (no mock data)
2. **File System Management**: Save, retrieve, and organize analysis results
3. **Context Persistence**: Maintain analysis continuity across sessions
4. **Task Planning**: Structure complex financial workflows

**File System Strategy:**
- **ALWAYS save important analysis results** to files for future reference
- **Use descriptive filenames** that indicate content and date
- **Organize data logically** in directories (stock_data/, reports/, research/)
- **Read existing files** before starting new analysis to avoid duplication
- **Update files** when new data becomes available

**File Organization Structure:**
```
/stock_data/          # Individual stock analysis and metrics
/portfolio_analysis/  # Portfolio performance and optimization
/market_research/     # Market trends and sector analysis  
/risk_reports/        # Risk assessments and stress tests
/strategies/          # Investment strategies and recommendations
/daily_notes/         # Daily market observations and notes
```

**Workflow for Complex Analysis:**
1. **Check existing files** with ls() to see what data is already available
2. **Read relevant files** to understand previous analysis
3. **Gather fresh data** using YFinance tools (always use real, current data)
4. **Create TODO list** for complex multi-step analysis
5. **Save results** to appropriate files as you complete each step
6. **Update existing files** when new data supersedes old analysis

**File Naming Conventions:**
- Stock analysis: "{SYMBOL}_analysis_{YYYYMMDD}.json"
- Portfolio reports: "portfolio_{name}_{YYYYMMDD}.json"
- Market research: "market_trends_{YYYYMMDD}.md"
- Risk reports: "risk_assessment_{YYYYMMDD}.json"

**Data Persistence Guidelines:**
- Save raw financial data for future reference
- Store calculated metrics and ratios
- Preserve analysis methodology and assumptions
- Keep historical snapshots for trend analysis
- Document data sources and timestamps

**Real Data Requirements:**
- ALWAYS use actual YFinance data, never mock or test data
- Fetch current market prices and real financial metrics
- Use historical data for backtesting and trend analysis
- Verify data freshness and market hours status
- Handle real market conditions and data availability

**Context Management:**
- Maintain continuity across long analysis sessions
- Reference previous findings when available
- Build cumulative knowledge base through file system
- Enable sophisticated multi-day research projects

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

Remember: The file system is your persistent memory. Use it extensively to build sophisticated, data-driven financial analysis that improves over time."""
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        prompt=system_message
    )
    
    return agent

def run_file_system_demo():
    """
    Demonstrate the file system agent with persistent context scenarios
    """
    print("💾 DeepAgent Financial Systems - File System Agent Demo")
    print("=" * 62)
    
    try:
        agent = create_file_system_agent()
        print("✓ File system agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create agent: {str(e)}")
        return
    
    # File system demonstration scenarios
    file_system_scenarios = [
        """Analyze Apple (AAPL) stock comprehensively and save the analysis. 
        Include current price, financial metrics, risk assessment, and your recommendation. 
        Save everything to a file so we can reference it later.""",
        
        """Check what stock analysis files we have saved, then read the Apple analysis.
        Based on that previous analysis, now compare Apple with Microsoft (MSFT) 
        and save a comparative analysis file.""",
        
        """Create a portfolio analysis for a $100,000 investment split between:
        - 40% Apple (AAPL)
        - 30% Microsoft (MSFT) 
        - 30% Google (GOOGL)
        
        Save the portfolio analysis and create a monitoring TODO list for ongoing tracking.""",
        
        """Look at our saved files, read any relevant previous analysis, then create
        a comprehensive market overview report. Include current market conditions,
        sector performance, and outlook. Save everything for future reference.""",
        
        """Based on all our previous saved analysis and files, create an investment
        strategy document. Consider our portfolio, market conditions, and individual
        stock analysis to recommend next steps."""
    ]
    
    print(f"\n📁 File System Integration Scenarios:")
    for i, scenario in enumerate(file_system_scenarios, 1):
        print(f"\n{i}. {scenario[:80]}...")
    
    # Interactive demo
    while True:
        print(f"\n" + "=" * 62)
        print("Options:")
        print("  1-5: Run file system scenario")
        print("  ls: List current files")
        print("  custom: Enter your own query")
        print("  quit: Exit demo")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == "quit":
            break
        elif choice == "ls":
            # Show current file structure
            print(ls("/"))
        elif choice == "custom":
            custom_query = input("Enter your query (will save results to files): ").strip()
            if custom_query:
                run_file_system_query(agent, custom_query)
        elif choice.isdigit() and 1 <= int(choice) <= len(file_system_scenarios):
            scenario_index = int(choice) - 1
            run_file_system_query(agent, file_system_scenarios[scenario_index])
        else:
            print("Invalid choice. Please try again.")
    
    print("\n👋 File system demo completed!")

def run_file_system_query(agent, query: str, session_id: str = "filesystem_session"):
    """
    Run a query with file system context management
    """
    config_dict = {"configurable": {"thread_id": session_id}}
    
    print(f"\n💾 File System Query:")
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
                        # Filter out file system tool output since it's already shown
                        if not (hasattr(latest_message, 'tool_calls') and latest_message.tool_calls):
                            print(f"\n{latest_message.content}")
                    
                    # Show non-file-system tool usage
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            if tool_call['name'] not in ['ls', 'read_file', 'write_file', 'edit_file']:
                                print(f"\n🔧 Using: {tool_call['name']} with real YFinance data")
        
        print(f"\n✅ File system workflow completed with persistent context!")
        
    except Exception as e:
        print(f"❌ Error in file system workflow: {str(e)}")

if __name__ == "__main__":
    run_file_system_demo()