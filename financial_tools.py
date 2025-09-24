"""
Financial Tools for DeepAgent Financial Systems
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from langchain_core.tools import tool
import json
import logging
import time
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting decorator
def rate_limit(calls_per_minute=4):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                print(f"⏳ Rate limiting: waiting {left_to_wait:.1f} seconds...")
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@tool
@rate_limit(calls_per_minute=4)
def get_stock_price(symbol: str, period: str = "1d") -> str:
    """
    Get current stock price using REAL YFinance market data.
    """
    try:
        print(f"🔍 Fetching real market data for {symbol.upper()}...")
        time.sleep(2)
        
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return json.dumps({
                "error": f"No market data available for {symbol}",
                "suggestion": "Check if the stock symbol is correct and markets are open",
                "symbol": symbol.upper(),
                "timestamp": datetime.now().isoformat()
            })
        
        current_price = float(hist['Close'].iloc[-1])
        prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
        
        result = {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "previous_close": round(prev_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "data_source": "YFinance - REAL Market Data",
            "last_updated": datetime.now().isoformat()
        }
        
        print(f"✅ Successfully retrieved real market data for {symbol.upper()}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting stock price for {symbol}: {str(e)}")
        return json.dumps({
            "error": f"Unable to retrieve market data for {symbol}",
            "error_details": str(e),
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat()
        }, indent=2)

@tool
def get_stock_history(symbol: str, period: str = "1y", interval: str = "1d") -> str:
    """Get historical stock data."""
    return json.dumps({"message": "Historical data temporarily disabled"})

@tool
def get_financial_statements(symbol: str, statement_type: str = "income") -> str:
    """Get financial statements."""
    return json.dumps({"message": "Financial statements temporarily disabled"})

@tool
def analyze_portfolio_performance(symbols: str, weights: str = None) -> str:
    """Analyze portfolio performance."""
    return json.dumps({"message": "Portfolio analysis temporarily disabled"})

@tool
def get_market_overview(market: str = "US") -> str:
    """Get market overview."""
    return json.dumps({"message": "Market overview temporarily disabled"})

@tool
def calculate_risk_metrics(symbol: str, benchmark: str = "^GSPC") -> str:
    """Calculate risk metrics."""
    return json.dumps({"message": "Risk metrics temporarily disabled"})

# Export all tools for use in agents
FINANCIAL_TOOLS = [
    get_stock_price,
    get_stock_history, 
    get_financial_statements,
    analyze_portfolio_performance,
    get_market_overview,
    calculate_risk_metrics
]
