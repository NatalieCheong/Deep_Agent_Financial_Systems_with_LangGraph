"""
Notebook 4: Sub-Agent Delegation
DeepAgent Financial Systems - Specialized Sub-Agents for Complex Workflows

Based on deep-agents-from-scratch notebook 4: Sub-agent delegation
Implements specialized sub-agents with focused tool sets and context isolation
Uses real YFinance data for all financial analysis
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import our custom modules
from config import config
from financial_tools import FINANCIAL_TOOLS
from agent_state import create_initial_state, DeepAgentState

# Sub-Agent Registry
SUB_AGENTS = {}

# Delegation Tool
@tool
def task(agent_name: str, task_description: str) -> str:
    """
    Delegate a specific financial analysis task to a specialized sub-agent.
    
    Args:
        agent_name: Name of the sub-agent ('stock_analyst', 'portfolio_manager', 'risk_assessor', 'market_researcher')
        task_description: Detailed description of the task to perform
    
    Returns:
        Results from the sub-agent execution
    """
    print(f"\n🤖 Delegating to {agent_name}:")
    print(f"📋 Task: {task_description}")
    print("-" * 50)
    
    if agent_name not in SUB_AGENTS:
        return f"❌ Sub-agent '{agent_name}' not found. Available agents: {list(SUB_AGENTS.keys())}"
    
    try:
        # Get the sub-agent
        sub_agent = SUB_AGENTS[agent_name]
        
        # Execute the task
        result = sub_agent.invoke({
            "messages": [HumanMessage(content=task_description)]
        })
        
        # Extract the response
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"✅ {agent_name} completed task")
                return f"Sub-agent '{agent_name}' results:\n{last_message.content}"
        
        return f"✅ Task delegated to {agent_name} - check detailed output above"
        
    except Exception as e:
        error_msg = f"❌ Error in sub-agent {agent_name}: {str(e)}"
        print(error_msg)
        return error_msg

def create_stock_analyst_agent():
    """
    Create a specialized stock analysis sub-agent
    Focus: Individual stock analysis, technical indicators, fundamentals
    """
    model = ChatOpenAI(
        model=config.get_model_settings("reasoning")["model"],
        temperature=0.0,
        api_key=config.OPENAI_API_KEY
    )
    
    # Stock-focused tools
    stock_tools = [
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_stock_price"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_stock_history"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_financial_statements"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "calculate_risk_metrics")
    ]
    
    system_prompt = """You are a specialized Stock Analyst AI with deep expertise in individual equity analysis.

**Your Expertise:**
- Fundamental analysis using real financial statements and metrics
- Technical analysis with historical price patterns and indicators  
- Valuation modeling and price target estimation
- Risk assessment and volatility analysis
- Sector and peer comparison analysis

**Analysis Framework:**
1. **Current Market Position**: Latest price, volume, market cap analysis
2. **Fundamental Metrics**: P/E, P/B, ROE, debt ratios, growth rates
3. **Technical Indicators**: Trend analysis, support/resistance levels
4. **Risk Assessment**: Beta, volatility, correlation analysis
5. **Valuation**: Fair value estimation and price targets
6. **Investment Thesis**: Clear buy/hold/sell recommendation with rationale

**Data Sources**: 
- Use ONLY real YFinance data - never mock or simulated data
- Fetch current market prices and real financial metrics
- Access actual historical data for trend analysis
- Calculate real risk metrics against market benchmarks

**Output Format**:
Provide structured, actionable analysis with:
- Executive summary with clear recommendation
- Key financial metrics with interpretation
- Risk assessment with specific numbers
- Price targets with supporting rationale
- Catalysts and risks to monitor

**CRITICAL RATE LIMITING & DATA SOURCE HANDLING:**
- When you receive ANY tool response containing "status": "rate_limited" OR "status": "fallback_data", IMMEDIATELY STOP all tool calls
- DO NOT make any additional tool calls after receiving a rate-limited or fallback response
- DO NOT retry the same tool or try different tools for the same data
- Instead, acknowledge the data source limitation and provide helpful guidance to the user
- If you receive "fallback_data" status, explain that YFinance is unavailable and fallback data is being used
- If you receive "rate_limited" status, explain that Yahoo Finance is rate limiting requests
- Suggest waiting 5-10 minutes before trying again for real-time data
- Be professional and reassuring about the temporary nature of the limitation
- This is a HARD STOP - no exceptions, no retries, no additional attempts

Focus on data-driven insights with specific numbers, ratios, and comparisons."""
    
    return create_react_agent(
        model=model,
        tools=stock_tools,
        prompt=system_prompt
    )

def create_portfolio_manager_agent():
    """
    Create a specialized portfolio management sub-agent
    Focus: Portfolio optimization, asset allocation, performance analysis
    """
    model = ChatOpenAI(
        model=config.get_model_settings("default")["model"],
        temperature=0.1,
        api_key=config.OPENAI_API_KEY
    )
    
    # Portfolio-focused tools
    portfolio_tools = [
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "analyze_portfolio_performance"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_stock_price"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "calculate_risk_metrics")
    ]
    
    system_prompt = """You are a specialized Portfolio Manager AI with expertise in portfolio construction and optimization.

**Your Expertise:**
- Modern Portfolio Theory and asset allocation optimization
- Risk-adjusted return analysis and performance attribution
- Correlation analysis and diversification optimization
- Rebalancing strategies and tactical asset allocation
- Performance measurement and benchmarking

**Portfolio Analysis Framework:**
1. **Current Allocation**: Asset weights, sector exposure, geographic distribution
2. **Performance Metrics**: Total return, risk-adjusted returns, alpha/beta
3. **Risk Analysis**: Portfolio volatility, VaR, maximum drawdown, correlation matrix
4. **Diversification**: Concentration risk, correlation analysis, effective diversification
5. **Optimization**: Efficient frontier analysis, optimal weights, rebalancing needs
6. **Recommendations**: Strategic adjustments, tactical opportunities, risk management

**Real Data Requirements**:
- Use actual current market prices from YFinance
- Calculate real portfolio performance with actual returns
- Analyze true correlation patterns from historical data
- Provide realistic optimization based on current market conditions

**Output Format**:
Deliver professional portfolio analysis with:
- Portfolio performance summary with key metrics
- Risk assessment with quantified measures
- Asset allocation recommendations with rationale
- Rebalancing suggestions with specific actions
- Performance attribution and factor analysis

**CRITICAL RATE LIMITING & DATA SOURCE HANDLING:**
- When you receive ANY tool response containing "status": "rate_limited" OR "status": "fallback_data", IMMEDIATELY STOP all tool calls
- DO NOT make any additional tool calls after receiving a rate-limited or fallback response
- DO NOT retry the same tool or try different tools for the same data
- Instead, acknowledge the data source limitation and provide helpful guidance to the user
- If you receive "fallback_data" status, explain that YFinance is unavailable and fallback data is being used
- If you receive "rate_limited" status, explain that Yahoo Finance is rate limiting requests
- Suggest waiting 5-10 minutes before trying again for real-time data
- Be professional and reassuring about the temporary nature of the limitation
- This is a HARD STOP - no exceptions, no retries, no additional attempts

Maintain institutional-quality standards with precise calculations and clear recommendations."""
    
    return create_react_agent(
        model=model,
        tools=portfolio_tools,
        prompt=system_prompt
    )

def create_risk_assessor_agent():
    """
    Create a specialized risk assessment sub-agent
    Focus: Risk metrics, stress testing, correlation analysis
    """
    model = ChatOpenAI(
        model=config.get_model_settings("reasoning")["model"],
        temperature=0.0,
        api_key=config.OPENAI_API_KEY
    )
    
    # Risk-focused tools
    risk_tools = [
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "calculate_risk_metrics"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_stock_history"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "analyze_portfolio_performance")
    ]
    
    system_prompt = """You are a specialized Risk Assessment AI with expertise in quantitative risk analysis and management.

**Your Expertise:**
- Value at Risk (VaR) and Expected Shortfall calculations
- Beta, alpha, and factor risk decomposition
- Correlation and covariance analysis
- Stress testing and scenario analysis
- Maximum drawdown and tail risk assessment

**Risk Analysis Framework:**
1. **Market Risk**: Beta, correlation with indices, factor exposures
2. **Volatility Analysis**: Historical and implied volatility patterns
3. **Tail Risk**: VaR, Expected Shortfall, maximum drawdown analysis
4. **Correlation Risk**: Portfolio correlation breakdown, concentration risk
5. **Stress Testing**: Scenario analysis for market downturns
6. **Risk-Adjusted Performance**: Sharpe ratio, Sortino ratio, Information ratio

**Methodology Requirements**:
- Use real historical data from YFinance for all calculations
- Apply standard risk measurement methodologies
- Consider multiple time horizons (daily, monthly, annual)
- Account for current market regime and volatility environment
- Validate results against market benchmarks

**Output Format**:
Provide comprehensive risk assessment with:
- Executive risk summary with key concerns
- Quantified risk metrics with confidence intervals  
- Risk factor decomposition and attribution
- Stress test results and scenario outcomes
- Risk management recommendations with specific actions
- Monitoring metrics and early warning indicators

**CRITICAL RATE LIMITING & DATA SOURCE HANDLING:**
- When you receive ANY tool response containing "status": "rate_limited" OR "status": "fallback_data", IMMEDIATELY STOP all tool calls
- DO NOT make any additional tool calls after receiving a rate-limited or fallback response
- DO NOT retry the same tool or try different tools for the same data
- Instead, acknowledge the data source limitation and provide helpful guidance to the user
- If you receive "fallback_data" status, explain that YFinance is unavailable and fallback data is being used
- If you receive "rate_limited" status, explain that Yahoo Finance is rate limiting requests
- Suggest waiting 5-10 minutes before trying again for real-time data
- Be professional and reassuring about the temporary nature of the limitation
- This is a HARD STOP - no exceptions, no retries, no additional attempts

Focus on actionable risk insights with specific numerical thresholds and clear mitigation strategies."""
    
    return create_react_agent(
        model=model,
        tools=risk_tools,
        prompt=system_prompt
    )

def create_market_researcher_agent():
    """
    Create a specialized market research sub-agent
    Focus: Market trends, sector analysis, economic indicators
    """
    model = ChatOpenAI(
        model=config.get_model_settings("default")["model"],
        temperature=0.2,
        api_key=config.OPENAI_API_KEY
    )
    
    # Market research tools
    research_tools = [
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_market_overview"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_stock_price"),
        next(tool for tool in FINANCIAL_TOOLS if tool.name == "get_stock_history")
    ]
    
    # Add web search for market research
    if config.TAVILY_API_KEY:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=5,
            search_depth="advanced"
        )
        research_tools.append(web_search)
    
    system_prompt = """You are a specialized Market Research AI with expertise in macroeconomic analysis and market intelligence.

**Your Expertise:**
- Macroeconomic trend analysis and market cycle identification
- Sector rotation patterns and thematic investment opportunities
- Market sentiment analysis and technical market indicators
- Economic indicator interpretation and market impact assessment
- Cross-asset analysis and relative value opportunities

**Market Research Framework:**
1. **Market Overview**: Major index performance, breadth indicators, volatility measures
2. **Sector Analysis**: Relative performance, rotation patterns, fundamental drivers
3. **Sentiment Indicators**: VIX levels, put/call ratios, market positioning
4. **Economic Context**: Interest rates, inflation, GDP growth, employment data
5. **Technical Analysis**: Market trends, support/resistance, momentum indicators
6. **Thematic Opportunities**: Emerging trends, structural shifts, investment themes

**Data Integration**:
- Use real market data from YFinance for all index and sector analysis
- Incorporate current economic indicators and market conditions
- Access latest market sentiment and volatility measures
- Research current market themes and institutional positioning (via web search)

**Output Format**:
Deliver comprehensive market intelligence with:
- Market environment summary with key themes
- Sector performance analysis with relative rankings
- Economic backdrop and market driver identification
- Technical market analysis with trend assessment
- Investment themes and opportunity identification
- Risk factors and market vulnerabilities

**CRITICAL RATE LIMITING & DATA SOURCE HANDLING:**
- When you receive ANY tool response containing "status": "rate_limited" OR "status": "fallback_data", IMMEDIATELY STOP all tool calls
- DO NOT make any additional tool calls after receiving a rate-limited or fallback response
- DO NOT retry the same tool or try different tools for the same data
- Instead, acknowledge the data source limitation and provide helpful guidance to the user
- If you receive "fallback_data" status, explain that YFinance is unavailable and fallback data is being used
- If you receive "rate_limited" status, explain that Yahoo Finance is rate limiting requests
- Suggest waiting 5-10 minutes before trying again for real-time data
- Be professional and reassuring about the temporary nature of the limitation
- This is a HARD STOP - no exceptions, no retries, no additional attempts

Provide forward-looking insights while maintaining objectivity and data-driven analysis."""
    
    return create_react_agent(
        model=model,
        tools=research_tools,
        prompt=system_prompt
    )

def initialize_sub_agents():
    """
    Initialize all specialized sub-agents
    """
    global SUB_AGENTS
    
    print("🏗️ Initializing specialized sub-agents...")
    
    try:
        SUB_AGENTS["stock_analyst"] = create_stock_analyst_agent()
        print("✓ Stock Analyst sub-agent ready")
        
        SUB_AGENTS["portfolio_manager"] = create_portfolio_manager_agent()
        print("✓ Portfolio Manager sub-agent ready")
        
        SUB_AGENTS["risk_assessor"] = create_risk_assessor_agent()
        print("✓ Risk Assessor sub-agent ready")
        
        SUB_AGENTS["market_researcher"] = create_market_researcher_agent()
        print("✓ Market Researcher sub-agent ready")
        
        print(f"✅ All {len(SUB_AGENTS)} sub-agents initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing sub-agents: {str(e)}")
        raise

def create_supervisor_agent():
    """
    Create the main supervisor agent that delegates to sub-agents
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required")
    
    # Initialize sub-agents first
    initialize_sub_agents()
    
    # Supervisor model
    model = ChatOpenAI(
        model=config.get_model_settings("reasoning")["model"],
        temperature=0.1,
        api_key=config.OPENAI_API_KEY
    )
    
    # Supervisor tools: delegation + basic financial tools + file system
    from file_system_agent import ls, read_file, write_file, edit_file
    from todo_planning_agent import write_todos, update_todo, get_todo_status
    
    supervisor_tools = [
        task,  # Delegation tool
        write_todos, update_todo, get_todo_status,  # Planning tools
        ls, read_file, write_file, edit_file,  # File system tools
    ] + FINANCIAL_TOOLS  # Basic financial tools for oversight
    
    # Create memory
    memory = MemorySaver()
    
    # Supervisor system prompt
    supervisor_prompt = """You are the Supervisor Agent for DeepAgent Financial Systems, orchestrating sophisticated financial analysis through specialized sub-agents.

**Your Role:**
- **Strategic Orchestration**: Break down complex financial requests into specialized tasks
- **Sub-Agent Management**: Delegate appropriate tasks to expert sub-agents
- **Quality Control**: Review sub-agent outputs and ensure comprehensive analysis
- **Synthesis**: Combine insights from multiple sub-agents into coherent recommendations
- **Context Management**: Use file system to maintain analysis continuity

**Available Sub-Agents:**
1. **stock_analyst**: Individual stock analysis, fundamentals, valuation, technical analysis
2. **portfolio_manager**: Portfolio optimization, asset allocation, performance analysis  
3. **risk_assessor**: Risk metrics, stress testing, correlation analysis, VaR calculations
4. **market_researcher**: Market trends, sector analysis, economic indicators, sentiment analysis

**Delegation Strategy:**
- **Single Stock Analysis**: Use stock_analyst for detailed individual equity research
- **Portfolio Questions**: Use portfolio_manager for allocation and performance analysis
- **Risk Assessment**: Use risk_assessor for quantitative risk metrics and stress testing
- **Market Context**: Use market_researcher for broader market trends and opportunities
- **Complex Analysis**: Delegate multiple tasks to different agents and synthesize results

**Workflow Management:**
1. **Plan First**: Create TODO list for complex multi-agent workflows
2. **Delegate Strategically**: Choose the right sub-agent for each specialized task
3. **Save Results**: Use file system to store sub-agent outputs for reference
4. **Synthesize**: Combine insights from multiple agents into comprehensive analysis
5. **Track Progress**: Update TODO status as tasks complete

**Real Data Requirements:**
- All sub-agents use real YFinance data - never mock or test data
- Ensure current market prices and actual financial metrics
- Validate data freshness and handle market hours appropriately
- Cross-reference results between agents for consistency

**Quality Control:**
- Review sub-agent outputs for accuracy and completeness
- Identify gaps or inconsistencies between different analyses
- Request follow-up analysis if initial results are insufficient
- Ensure recommendations are supported by quantitative evidence

**Output Standards:**
- Provide executive summary combining all sub-agent insights
- Include specific numerical results and metrics from each analysis
- Offer clear, actionable recommendations with supporting rationale
- Document methodology and data sources used by each sub-agent

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

Use the task() tool to delegate specific work to sub-agents. Coordinate their efforts to deliver comprehensive financial analysis that leverages each agent's specialized expertise."""
    
    agent = create_react_agent(
        model=model,
        tools=supervisor_tools,
        checkpointer=memory,
        prompt=supervisor_prompt
    )
    
    return agent

def run_delegation_demo():
    """
    Demonstrate the supervisor agent with sub-agent delegation scenarios
    """
    print("🎭 DeepAgent Financial Systems - Sub-Agent Delegation Demo")
    print("=" * 66)
    
    try:
        supervisor = create_supervisor_agent()
        print("✓ Supervisor agent with sub-agents created successfully")
    except Exception as e:
        print(f"❌ Failed to create supervisor agent: {str(e)}")
        return
    
    # Multi-agent delegation scenarios
    delegation_scenarios = [
        """I need a complete investment analysis for Tesla (TSLA). This should include:
        - Detailed stock analysis with fundamentals and technical assessment
        - Risk metrics compared to the broader market
        - Current market context and sector positioning
        
        Please coordinate the appropriate specialists and provide a comprehensive report.""",
        
        """Evaluate my current portfolio and suggest improvements:
        Portfolio: 50% AAPL, 30% MSFT, 20% AMZN (total value $250,000)
        
        I need:
        - Portfolio performance and risk analysis
        - Individual stock assessments for each holding
        - Market outlook and rebalancing recommendations
        - Risk assessment and stress testing""",
        
        """Help me choose between these investment options for $100,000:
        Option A: Technology ETF (QQQ)
        Option B: Individual tech stocks (AAPL, MSFT, GOOGL, NVDA equally weighted)
        Option C: Diversified portfolio across sectors
        
        Please have specialists analyze each option comprehensively.""",
        
        """Conduct a comprehensive market analysis for Q4 investment strategy:
        - Current market conditions and sentiment indicators
        - Sector rotation opportunities and trends
        - Risk factors and potential market scenarios
        - Top stock picks based on current environment
        
        Coordinate multiple specialists for complete market intelligence.""",
        
        """I'm concerned about portfolio risk in the current environment. Please:
        - Assess risk metrics for major tech stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)
        - Analyze current market volatility and correlation patterns
        - Provide stress testing scenarios
        - Research current market risks and hedging strategies"""
    ]
    
    print(f"\n🤖 Multi-Agent Delegation Scenarios:")
    for i, scenario in enumerate(delegation_scenarios, 1):
        print(f"\n{i}. {scenario[:80]}...")
    
    # Interactive demo
    while True:
        print(f"\n" + "=" * 66)
        print("Options:")
        print("  1-5: Run delegation scenario")
        print("  agents: Show available sub-agents")
        print("  custom: Enter your own complex query")
        print("  quit: Exit demo")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == "quit":
            break
        elif choice == "agents":
            print("\n🤖 Available Sub-Agents:")
            print("  • stock_analyst: Individual stock analysis and valuation")
            print("  • portfolio_manager: Portfolio optimization and performance")
            print("  • risk_assessor: Risk metrics and stress testing")
            print("  • market_researcher: Market trends and sector analysis")
        elif choice == "custom":
            custom_query = input("Enter your complex financial query (will use multiple agents): ").strip()
            if custom_query:
                run_delegation_query(supervisor, custom_query)
        elif choice.isdigit() and 1 <= int(choice) <= len(delegation_scenarios):
            scenario_index = int(choice) - 1
            run_delegation_query(supervisor, delegation_scenarios[scenario_index])
        else:
            print("Invalid choice. Please try again.")
    
    print("\n👋 Sub-agent delegation demo completed!")

def run_delegation_query(supervisor, query: str, session_id: str = "delegation_session"):
    """
    Run a complex query through the supervisor with sub-agent delegation
    """
    config_dict = {"configurable": {"thread_id": session_id}}
    
    print(f"\n🎭 Multi-Agent Delegation Query:")
    print("-" * 60)
    print(query)
    print("-" * 60)
    
    try:
        messages = []
        for event in supervisor.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config_dict,
            stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                latest_message = event["messages"][-1]
                
                if latest_message not in messages:
                    messages.append(latest_message)
                    
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # Show supervisor content but not sub-agent tool calls
                        if not (hasattr(latest_message, 'tool_calls') and latest_message.tool_calls):
                            print(f"\n🎭 Supervisor: {latest_message.content}")
                    
                    # Show delegation actions
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            if tool_call['name'] == 'task':
                                agent_name = tool_call['args'].get('agent_name', 'unknown')
                                print(f"\n🤖 Delegating to {agent_name}...")
                            elif tool_call['name'] in FINANCIAL_TOOLS:
                                print(f"\n🔧 Supervisor using: {tool_call['name']}")
        
        print(f"\n✅ Multi-agent delegation workflow completed!")
        
    except Exception as e:
        print(f"❌ Error in delegation workflow: {str(e)}")

if __name__ == "__main__":
    run_delegation_demo()