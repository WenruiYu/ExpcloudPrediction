#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ExpcloudPrediction main entry point.

This script serves as the central entry point for all functionality, including:
- Data collection: stock data and macroeconomic indicators
- Data cleaning: removing data files
- Feature engineering: creating and managing features
- Feature pipeline: generating model-ready data
"""

import sys
import logging
from pathlib import Path

# Configure base directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import CLI modules
from src.cli import data_cli
from src.cli import feature_cli
from src.core.config import LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def show_usage():
    """Show usage information."""
    print("ExpcloudPrediction: A Python-based application for financial data collection and analysis")
    print("\nAvailable commands:")
    print("  data     - Collect and process stock and macro data")
    print("  features - Generate features and prepare model inputs")
    print("  clean    - Clean data files")
    print("  all      - Run the complete pipeline (data collection + feature generation)")
    print("\nUsage examples:")
    print("  python -m src.main data --stock --symbol sh.600519")
    print("  python -m src.main features --symbol sh.600519 --include-time --include-lag")
    print("  python -m src.main clean --all")
    print("  python -m src.main all --symbol sh.600519")
    print("\nFor detailed help on specific commands:")
    print("  python -m src.main data --help")
    print("  python -m src.main features --help")
    print("  python -m src.main clean --help")

def main():
    """Main entry point function."""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1]
    
    # Remove the command from sys.argv to allow argparse to work correctly
    sys.argv.pop(1)
    
    if command == "data":
        logger.info("Running data collection CLI")
        data_cli.main()
    elif command == "features":
        logger.info("Running feature pipeline CLI")
        feature_cli.main()
    elif command == "clean":
        # Import the clean_data module and run its main function
        from src.data import cleaner
        sys.argv[0] = "python -m src.data.cleaner"
        logger.info("Running data cleaning CLI")
        cleaner.main()
    elif command == "all":
        logger.info("Running complete pipeline")
        # First collect data
        data_cli.main()
        # Then generate features
        feature_cli.main()
    else:
        print(f"Unknown command: {command}")
        show_usage()

if __name__ == "__main__":
    main() 