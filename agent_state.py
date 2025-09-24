"""
Agent State Management for DeepAgent Financial Systems
Based on LangChain deep-agents-from-scratch patterns with financial data focus
"""

from typing import Annotated, Dict, List, Optional, Any, TypedDict, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from datetime import datetime
from datetime import timedelta
import json

# TODO Task Status Types
TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]

class TodoItem(TypedDict):
    """Individual TODO item for task planning"""
    id: str
    task: str
    status: TodoStatus
    created_at: str
    updated_at: str
    priority: Literal["low", "medium", "high", "urgent"]
    assigned_agent: Optional[str]
    dependencies: List[str]
    metadata: Dict[str, Any]

class FileSystemItem(TypedDict):
    """Virtual file system item"""
    name: str
    content: str
    type: Literal["file", "directory"]
    created_at: str
    updated_at: str
    size: int
    metadata: Dict[str, Any]

class FinancialDataCache(TypedDict):
    """Cache for financial data to avoid redundant API calls"""
    symbol: str
    data_type: str
    data: Dict[str, Any]
    timestamp: str
    expiry: str

class SubAgentInfo(TypedDict):
    """Information about sub-agents"""
    name: str
    description: str
    tools: List[str]
    status: Literal["idle", "working", "completed", "error"]
    created_at: str
    last_active: str
    results: Optional[Dict[str, Any]]

class DeepAgentState(TypedDict):
    """
    Main state for DeepAgent Financial Systems
    Incorporates all patterns from deep-agents-from-scratch
    """
    # Core messaging
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Task Planning (from notebook 2)
    todos: List[TodoItem]
    current_task: Optional[str]
    planning_context: Dict[str, Any]
    
    # Virtual File System (from notebook 3)
    files: Dict[str, FileSystemItem]
    current_directory: str
    file_operations_log: List[Dict[str, Any]]
    
    # Sub-Agent Management (from notebook 4)
    sub_agents: Dict[str, SubAgentInfo]
    active_sub_agents: List[str]
    delegation_history: List[Dict[str, Any]]
    
    # Financial Data Management
    financial_cache: Dict[str, FinancialDataCache]
    watchlist: List[str]
    portfolio: Dict[str, Dict[str, Any]]
    market_data: Dict[str, Any]
    
    # Research and Analysis (from notebook 5)
    research_context: Dict[str, Any]
    analysis_results: List[Dict[str, Any]]
    web_search_results: List[Dict[str, Any]]
    
    # Agent Control
    iteration_count: int
    max_iterations: int
    start_time: str
    last_activity: str
    agent_status: Literal["initializing", "planning", "executing", "delegating", "summarizing", "completed", "error"]
    
    # Session Management
    session_id: str
    user_preferences: Dict[str, Any]
    conversation_context: Dict[str, Any]

# Helper functions for state management

def create_initial_state(
    user_message: str,
    session_id: str = None,
    max_iterations: int = 50
) -> DeepAgentState:
    """Create initial agent state"""
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    now = datetime.now().isoformat()
    
    return DeepAgentState(
        # Core messaging
        messages=[HumanMessage(content=user_message)],
        
        # Task Planning
        todos=[],
        current_task=None,
        planning_context={},
        
        # Virtual File System
        files={},
        current_directory="/",
        file_operations_log=[],
        
        # Sub-Agent Management
        sub_agents={},
        active_sub_agents=[],
        delegation_history=[],
        
        # Financial Data Management
        financial_cache={},
        watchlist=[],
        portfolio={},
        market_data={},
        
        # Research and Analysis
        research_context={},
        analysis_results=[],
        web_search_results=[],
        
        # Agent Control
        iteration_count=0,
        max_iterations=max_iterations,
        start_time=now,
        last_activity=now,
        agent_status="initializing",
        
        # Session Management
        session_id=session_id,
        user_preferences={},
        conversation_context={}
    )

def add_todo_item(
    state: DeepAgentState,
    task: str,
    priority: Literal["low", "medium", "high", "urgent"] = "medium",
    assigned_agent: Optional[str] = None,
    dependencies: List[str] = None
) -> DeepAgentState:
    """Add a new TODO item to the state"""
    if dependencies is None:
        dependencies = []
    
    now = datetime.now().isoformat()
    todo_id = f"todo_{len(state['todos']) + 1}_{datetime.now().strftime('%H%M%S')}"
    
    new_todo = TodoItem(
        id=todo_id,
        task=task,
        status="pending",
        created_at=now,
        updated_at=now,
        priority=priority,
        assigned_agent=assigned_agent,
        dependencies=dependencies,
        metadata={}
    )
    
    state["todos"].append(new_todo)
    state["last_activity"] = now
    return state

def update_todo_status(
    state: DeepAgentState,
    todo_id: str,
    status: TodoStatus,
    metadata: Dict[str, Any] = None
) -> DeepAgentState:
    """Update the status of a TODO item"""
    now = datetime.now().isoformat()
    
    for todo in state["todos"]:
        if todo["id"] == todo_id:
            todo["status"] = status
            todo["updated_at"] = now
            if metadata:
                todo["metadata"].update(metadata)
            break
    
    state["last_activity"] = now
    return state

def write_file(
    state: DeepAgentState,
    filename: str,
    content: str,
    file_type: Literal["file", "directory"] = "file"
) -> DeepAgentState:
    """Write content to virtual file system"""
    now = datetime.now().isoformat()
    
    file_item = FileSystemItem(
        name=filename,
        content=content,
        type=file_type,
        created_at=now,
        updated_at=now,
        size=len(content),
        metadata={}
    )
    
    state["files"][filename] = file_item
    
    # Log the operation
    operation = {
        "operation": "write",
        "filename": filename,
        "timestamp": now,
        "size": len(content)
    }
    state["file_operations_log"].append(operation)
    state["last_activity"] = now
    
    return state

def read_file(state: DeepAgentState, filename: str) -> Optional[str]:
    """Read content from virtual file system"""
    file_item = state["files"].get(filename)
    if file_item:
        # Log the operation
        now = datetime.now().isoformat()
        operation = {
            "operation": "read",
            "filename": filename,
            "timestamp": now
        }
        state["file_operations_log"].append(operation)
        return file_item["content"]
    return None

def list_files(state: DeepAgentState, directory: str = None) -> List[str]:
    """List files in the virtual file system"""
    if directory is None:
        directory = state["current_directory"]
    
    # Simple implementation - return all files for now
    # In a more sophisticated version, you'd implement proper directory structure
    return list(state["files"].keys())

def cache_financial_data(
    state: DeepAgentState,
    symbol: str,
    data_type: str,
    data: Dict[str, Any],
    expiry_minutes: int = 15
) -> DeepAgentState:
    """Cache financial data to avoid redundant API calls"""
    now = datetime.now()
    expiry = now + timedelta(minutes=expiry_minutes)
    
    cache_key = f"{symbol}_{data_type}"
    cache_item = FinancialDataCache(
        symbol=symbol,
        data_type=data_type,
        data=data,
        timestamp=now.isoformat(),
        expiry=expiry.isoformat()
    )
    
    state["financial_cache"][cache_key] = cache_item
    state["last_activity"] = now.isoformat()
    return state

def get_cached_financial_data(
    state: DeepAgentState,
    symbol: str,
    data_type: str
) -> Optional[Dict[str, Any]]:
    """Retrieve cached financial data if not expired"""
    cache_key = f"{symbol}_{data_type}"
    cache_item = state["financial_cache"].get(cache_key)
    
    if cache_item:
        now = datetime.now()
        expiry = datetime.fromisoformat(cache_item["expiry"])
        if now < expiry:
            return cache_item["data"]
        else:
            # Remove expired cache
            del state["financial_cache"][cache_key]
    
    return None

def register_sub_agent(
    state: DeepAgentState,
    name: str,
    description: str,
    tools: List[str]
) -> DeepAgentState:
    """Register a new sub-agent"""
    now = datetime.now().isoformat()
    
    sub_agent = SubAgentInfo(
        name=name,
        description=description,
        tools=tools,
        status="idle",
        created_at=now,
        last_active=now,
        results=None
    )
    
    state["sub_agents"][name] = sub_agent
    state["last_activity"] = now
    return state

def update_sub_agent_status(
    state: DeepAgentState,
    agent_name: str,
    status: Literal["idle", "working", "completed", "error"],
    results: Optional[Dict[str, Any]] = None
) -> DeepAgentState:
    """Update sub-agent status and results"""
    now = datetime.now().isoformat()
    
    if agent_name in state["sub_agents"]:
        state["sub_agents"][agent_name]["status"] = status
        state["sub_agents"][agent_name]["last_active"] = now
        if results:
            state["sub_agents"][agent_name]["results"] = results
    
    state["last_activity"] = now
    return state

def get_state_summary(state: DeepAgentState) -> Dict[str, Any]:
    """Get a summary of the current state for debugging/monitoring"""
    return {
        "session_id": state["session_id"],
        "agent_status": state["agent_status"],
        "iteration_count": state["iteration_count"],
        "max_iterations": state["max_iterations"],
        "message_count": len(state["messages"]),
        "todo_count": len(state["todos"]),
        "pending_todos": len([t for t in state["todos"] if t["status"] == "pending"]),
        "active_todos": len([t for t in state["todos"] if t["status"] == "in_progress"]),
        "file_count": len(state["files"]),
        "sub_agent_count": len(state["sub_agents"]),
        "active_sub_agents": len(state["active_sub_agents"]),
        "cached_data_count": len(state["financial_cache"]),
        "watchlist_size": len(state["watchlist"]),
        "portfolio_positions": len(state["portfolio"]),
        "last_activity": state["last_activity"]
    }

# State validation functions
def validate_state(state: DeepAgentState) -> List[str]:
    """Validate state consistency and return any issues"""
    issues = []
    
    # Check iteration limits
    if state["iteration_count"] >= state["max_iterations"]:
        issues.append(f"Iteration limit reached: {state['iteration_count']}/{state['max_iterations']}")
    
    # Check for orphaned dependencies in TODOs
    todo_ids = {todo["id"] for todo in state["todos"]}
    for todo in state["todos"]:
        for dep_id in todo["dependencies"]:
            if dep_id not in todo_ids:
                issues.append(f"TODO {todo['id']} has orphaned dependency: {dep_id}")
    
    # Check for inactive sub-agents that are marked as active
    for agent_name in state["active_sub_agents"]:
        if agent_name not in state["sub_agents"]:
            issues.append(f"Active sub-agent {agent_name} not found in sub_agents registry")
    
    return issues

