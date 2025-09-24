"""
DeepAgent Financial Systems - Main Application
Integrates all 5 progressive notebooks into a unified demonstration

Run this file to experience the complete DeepAgent Financial Systems capability
from basic ReAct agents to sophisticated multi-agent financial research
"""

import os
import sys
from datetime import datetime
from config import config

def print_banner():
    """Print the main application banner"""
    print("=" * 80)
    print("                      DEEPAGENT FINANCIAL SYSTEMS")
    print("                   Advanced AI Financial Research Platform")
    print("                     Built with LangGraph & OpenAI")
    print("=" * 80)
    print(f"🗓️  Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊  Data Source: Real-time YFinance Market Data")
    print(f"🤖  AI Models: OpenAI GPT-4 & GPT-4-mini")
    print(f"🔧  Framework: LangChain + LangGraph")
    print("=" * 80)

def check_configuration():
    """Check and display system configuration"""
    print("\n🔧 SYSTEM CONFIGURATION CHECK:")
    print("-" * 40)
    
    status = config.validate_config()
    
    # Required configurations
    if status["openai_configured"]:
        print("✅ OpenAI API Key: Configured")
    else:
        print("❌ OpenAI API Key: Missing (Required)")
        return False
    
    # Optional but recommended configurations
    if status["tavily_configured"]:
        print("✅ Tavily API Key: Configured (Web search enabled)")
    else:
        print("⚠️  Tavily API Key: Missing (Web search disabled)")
    
    if status["langsmith_configured"]:
        print("✅ LangSmith API Key: Configured (Tracing enabled)")
    else:
        print("⚠️  LangSmith API Key: Missing (No tracing)")
    
    print(f"📈 Default Model: {config.DEFAULT_MODEL}")
    print(f"🧠 Reasoning Model: {config.REASONING_MODEL}")
    print(f"⚡ Fast Model: {config.FAST_MODEL}")
    
    return True

def show_capabilities():
    """Display the full system capabilities"""
    print("\n🚀 DEEPAGENT FINANCIAL SYSTEMS CAPABILITIES:")
    print("-" * 50)
    
    print("\n📊 REAL-TIME FINANCIAL DATA:")
    print("  • Live stock prices and market data via YFinance")
    print("  • Historical price analysis and technical indicators")
    print("  • Financial statements and fundamental metrics")
    print("  • Portfolio performance and risk analytics")
    print("  • Market overview and sector analysis")
    print("  • Risk metrics: Beta, Alpha, Sharpe ratio, VaR")
    
    print("\n🤖 SPECIALIZED AI AGENTS:")
    print("  • Stock Analyst: Individual equity research & valuation")
    print("  • Portfolio Manager: Asset allocation & optimization")
    print("  • Risk Assessor: Quantitative risk analysis & stress testing")
    print("  • Market Researcher: Market trends & sector intelligence")
    
    print("\n🧠 ADVANCED CAPABILITIES:")
    print("  • Multi-agent orchestration and task delegation")
    print("  • Persistent context through virtual file system")
    print("  • Strategic planning with TODO workflow management")
    print("  • Web search integration for market intelligence")
    print("  • Professional research report generation")
    
    print("\n📋 PROGRESSIVE LEARNING MODULES:")
    print("  1. Basic Agent: ReAct loops with financial tools")
    print("  2. Planning Agent: TODO-based task management")
    print("  3. File System Agent: Context persistence & memory")
    print("  4. Sub-Agent Delegation: Specialized expert coordination")
    print("  5. Deep Research Agent: Complete institutional research")

def main_menu():
    """Display and handle the main menu"""
    while True:
        print("\n" + "=" * 80)
        print("                           MAIN MENU")
        print("=" * 80)
        print("Choose your DeepAgent Financial Systems experience:")
        print()
        print("📚 PROGRESSIVE LEARNING MODULES:")
        print("  1. Basic Financial Agent       - Start here: ReAct agent with financial tools")
        print("  2. TODO Planning Agent        - Add task planning and workflow management")  
        print("  3. File System Agent          - Add persistent context and memory")
        print("  4. Sub-Agent Delegation       - Add specialized expert coordination")
        print("  5. Deep Research Agent        - Complete institutional research platform")
        print()
        print("🚀 QUICK START:")
        print("  quick  - Jump directly to Deep Research Agent (full capabilities)")
        print("  demo   - Run guided demo with sample financial scenarios")
        print()
        print("ℹ️  INFORMATION:")
        print("  config - Show system configuration")
        print("  help   - Show detailed help and capabilities")
        print("  quit   - Exit application")
        print()
        
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == "quit" or choice == "q":
            print("\n👋 Thank you for using DeepAgent Financial Systems!")
            print("⭐ If you found this useful, please star the repository on GitHub!")
            break
        elif choice == "1":
            run_basic_agent()
        elif choice == "2":
            run_planning_agent()
        elif choice == "3":
            run_file_system_agent()
        elif choice == "4":
            run_delegation_agent()
        elif choice == "5":
            run_deep_research_agent()
        elif choice == "quick":
            print("\n🚀 Launching Deep Research Agent with full capabilities...")
            run_deep_research_agent()
        elif choice == "demo":
            run_guided_demo()
        elif choice == "config":
            check_configuration()
        elif choice == "help":
            show_capabilities()
        else:
            print("❌ Invalid choice. Please try again.")

def run_basic_agent():
    """Run the basic financial agent demo"""
    print("\n🤖 Starting Basic Financial Agent...")
    try:
        from basic_agent import demo_basic_financial_agent
        demo_basic_financial_agent()
    except ImportError as e:
        print(f"❌ Error importing basic agent: {e}")
    except Exception as e:
        print(f"❌ Error running basic agent: {e}")

def run_planning_agent():
    """Run the planning agent demo"""
    print("\n📋 Starting TODO Planning Agent...")
    try:
        from todo_planning_agent import run_planning_demo
        run_planning_demo()
    except ImportError as e:
        print(f"❌ Error importing planning agent: {e}")
    except Exception as e:
        print(f"❌ Error running planning agent: {e}")

def run_file_system_agent():
    """Run the file system agent demo"""
    print("\n💾 Starting File System Agent...")
    try:
        from file_system_agent import run_file_system_demo
        run_file_system_demo()
    except ImportError as e:
        print(f"❌ Error importing file system agent: {e}")
    except Exception as e:
        print(f"❌ Error running file system agent: {e}")

def run_delegation_agent():
    """Run the sub-agent delegation demo"""
    print("\n🎭 Starting Sub-Agent Delegation...")
    try:
        from sub_agent_delegation import run_delegation_demo
        run_delegation_demo()
    except ImportError as e:
        print(f"❌ Error importing delegation agent: {e}")
    except Exception as e:
        print(f"❌ Error running delegation agent: {e}")

def run_deep_research_agent():
    """Run the deep research agent demo"""
    print("\n🎓 Starting Deep Research Agent...")
    try:
        from deep_research_agent import run_deep_research_demo
        run_deep_research_demo()
    except ImportError as e:
        print(f"❌ Error importing deep research agent: {e}")
    except Exception as e:
        print(f"❌ Error running deep research agent: {e}")

def run_guided_demo():
    """Run a guided demo with sample scenarios"""
    print("\n🎯 GUIDED DEMO - DeepAgent Financial Systems")
    print("-" * 50)
    print("This demo will showcase key capabilities with real market data.")
    print()
    
    demo_scenarios = [
        "Analyze Apple (AAPL) stock with current market data",
        "Create a technology portfolio with risk assessment", 
        "Compare investment options: ETF vs individual stocks",
        "Research current market conditions and outlook"
    ]
    
    print("Available demo scenarios:")
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"  {i}. {scenario}")
    
    try:
        choice = input(f"\nChoose scenario (1-{len(demo_scenarios)}) or 'all' for complete demo: ").strip()
        
        if choice.lower() == "all":
            print("\n🚀 Running complete guided demo...")
            # Run through all scenarios with the deep research agent
            run_deep_research_agent()
        elif choice.isdigit() and 1 <= int(choice) <= len(demo_scenarios):
            scenario_index = int(choice) - 1
            print(f"\n🎯 Running scenario: {demo_scenarios[scenario_index]}")
            run_deep_research_agent()
        else:
            print("❌ Invalid choice.")
    except Exception as e:
        print(f"❌ Error in guided demo: {e}")

def main():
    """Main application entry point"""
    # Clear screen and show banner
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()
    
    # Check configuration
    if not check_configuration():
        print("\n❌ Configuration issues detected. Please check your .env file.")
        print("📖 Refer to .env.example for required API keys.")
        return
    
    # Show capabilities overview
    show_capabilities()
    
    # Start main menu
    main_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 DeepAgent Financial Systems session ended by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")