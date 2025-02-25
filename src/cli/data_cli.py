# src/cli/data_cli.py

import os
import argparse
import logging
import logging.config
import pandas as pd
from typing import Dict, Any, Optional, List

# Try to import using relative imports if running as a module
try:
    from src.core.config import (
        LOGGING_CONFIG, START_DATE, END_DATE,
        DEFAULT_TICKER_BS, MACRO_DATA_SOURCES
    )
    from src.data.stock_collection import StockDataCollector
    from src.data.macro_collection import MacroDataCollector
    from src.core.utils import IndicatorCalculator
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import (
        LOGGING_CONFIG, START_DATE, END_DATE,
        DEFAULT_TICKER_BS, MACRO_DATA_SOURCES
    )
    from ..data.stock_collection import StockDataCollector
    from ..data.macro_collection import MacroDataCollector
    from ..core.utils import IndicatorCalculator

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def collect_stock_data(
    symbol: str = DEFAULT_TICKER_BS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    force_refresh: bool = False,
    indicators: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Collect and process stock data.
    
    Args:
        symbol: Stock symbol to collect
        start_date: Start date for data collection
        end_date: End date for data collection
        force_refresh: Whether to force refresh data
        indicators: List of technical indicators to calculate
        
    Returns:
        Processed DataFrame or None on failure
    """
    logger.info(f"Collecting stock data for {symbol} from {start_date} to {end_date}")
    
    try:
        with StockDataCollector(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            indicators=indicators
        ) as collector:
            # Get data with optional forced refresh
            data = collector.get_data(force_refresh=force_refresh)
            
            if data is None:
                logger.error("Failed to collect stock data")
                return None
            
            logger.info(f"Successfully collected stock data: {len(data)} rows")
            return data
            
    except Exception as e:
        logger.error(f"Error collecting stock data: {e}", exc_info=True)
        return None

def collect_macro_data(
    sources: Optional[List[str]] = None,
    force_refresh: bool = False
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Collect and process macroeconomic data.
    
    Args:
        sources: List of macro data sources to collect
        force_refresh: Whether to force refresh data
        
    Returns:
        Dictionary of source names to their DataFrames
    """
    logger.info(f"Collecting macroeconomic data for sources: {sources or 'all'}")
    
    try:
        collector = MacroDataCollector(sources=sources)
        results = collector.fetch_all_sources(overwrite=force_refresh)
        
        # Log results summary
        success_count = sum(1 for data in results.values() if data is not None)
        logger.info(f"Successfully collected {success_count}/{len(results)} macro data sources")
        
        return results
        
    except Exception as e:
        logger.error(f"Error collecting macro data: {e}", exc_info=True)
        return {}

def run_data_collection(args: argparse.Namespace) -> None:
    """
    Run data collection based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    # Handle 'all' option for indicators
    if args.indicators and 'all' in args.indicators:
        all_indicators = [
            "sma", "ema", "rsi", "macd", "bollinger", "atr", "stochastic",
            "obv", "adx", "cci", "mfi", "williams_r", "psar", "ichimoku"
        ]
        args.indicators = all_indicators
        logger.info(f"Using all available indicators: {', '.join(all_indicators)}")
    
    # Collect stock data if requested
    if args.stock:
        stock_data = collect_stock_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
            indicators=args.indicators
        )
        
        if stock_data is not None:
            print(f"Stock data collection successful for {args.symbol}")
            print(f"Shape: {stock_data.shape}")
            print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
            print(f"Columns: {stock_data.columns.tolist()}")
            if not args.no_preview:
                print("\nData Preview:")
                print(stock_data.head())
    
    # Collect macro data if requested
    if args.macro:
        macro_data = collect_macro_data(
            sources=args.macro_sources,
            force_refresh=args.force_refresh
        )
        
        if macro_data:
            print("\nMacroeconomic Data Collection Results:")
            for source, data in macro_data.items():
                status = "Success" if data is not None else "Failed"
                rows = len(data) if data is not None else 0
                print(f"{source}: {status} ({rows} rows)")
                
                if data is not None and not args.no_preview:
                    print(f"\nPreview of {source} data:")
                    print(data.head())
                    print("\n")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="ExpcloudPrediction: Collect and process financial data."
    )
    
    # Stock data arguments
    parser.add_argument(
        "--stock",
        action="store_true",
        help="Collect stock data"
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_TICKER_BS,
        help=f"Stock symbol to collect (default: {DEFAULT_TICKER_BS})"
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"Start date for data collection (default: {START_DATE})"
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help=f"End date for data collection (default: {END_DATE})"
    )
    parser.add_argument(
        "--indicators",
        type=str,
        nargs="+",
        choices=[
            "sma", "ema", "rsi", "macd", "bollinger", "atr", "stochastic",
            "obv", "adx", "cci", "mfi", "williams_r", "psar", "ichimoku", "all"
        ],
        default=["sma", "ema", "rsi", "macd"],
        help="Technical indicators to calculate (default: sma ema rsi macd). Use 'all' for all indicators."
    )
    
    # Macro data arguments
    parser.add_argument(
        "--macro",
        action="store_true",
        help="Collect macroeconomic data"
    )
    parser.add_argument(
        "--macro-sources",
        nargs="+",
        choices=list(MACRO_DATA_SOURCES.keys()),
        help="Specific macro data sources to collect (default: all)"
    )
    
    # General arguments
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh data (ignore cache)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Don't show data preview"
    )
    
    args = parser.parse_args()
    
    # If no specific action is selected, collect both types of data
    if not args.stock and not args.macro:
        args.stock = True
        args.macro = True
    
    # Run data collection
    run_data_collection(args)

if __name__ == "__main__":
    main() 