#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Pipeline Script

This script demonstrates how to use the new data organization components together
to create a complete feature generation pipeline for the stock prediction model.
It handles:
1. Data collection (stock and macro)
2. Feature generation
3. Feature alignment and transformation
4. Model input preparation
"""

import os
import argparse
import logging
import logging.config
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Try to import using relative imports if running as a module
try:
    from src.core.config import (
        LOGGING_CONFIG, DEFAULT_TICKER_BS, START_DATE, END_DATE, DATA_DIR,
        MACRO_DATA_SOURCES
    )
    from src.features.integration import DataIntegrator
    from src.features.metadata import (
        FREQUENCY_DAILY, FREQUENCY_WEEKLY, FREQUENCY_MONTHLY,
        CATEGORY_STOCK, CATEGORY_MACRO
    )
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import (
        LOGGING_CONFIG, DEFAULT_TICKER_BS, START_DATE, END_DATE, DATA_DIR,
        MACRO_DATA_SOURCES
    )
    from ..features.integration import DataIntegrator
    from ..features.metadata import (
        FREQUENCY_DAILY, FREQUENCY_WEEKLY, FREQUENCY_MONTHLY,
        CATEGORY_STOCK, CATEGORY_MACRO
    )

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Pipeline for Stock Prediction"
    )
    
    # Data collection arguments
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
        "--macro-sources",
        nargs="+",
        choices=list(MACRO_DATA_SOURCES.keys()),
        default=['gdp', 'cpi', 'm2'],
        help="Macro data sources to include (default: gdp cpi m2)"
    )
    
    # Feature generation arguments
    parser.add_argument(
        "--frequency",
        choices=['daily', 'weekly', 'monthly'],
        default='daily',
        help="Target frequency for features (default: daily)"
    )
    parser.add_argument(
        "--include-time",
        action="store_true",
        help="Include time-based features"
    )
    parser.add_argument(
        "--include-lag",
        action="store_true",
        help="Include lag features"
    )
    parser.add_argument(
        "--include-rolling",
        action="store_true",
        help="Include rolling window features"
    )
    parser.add_argument(
        "--lag-periods",
        nargs="+",
        type=int,
        help="Lag periods to use (default: auto-selected based on frequency)"
    )
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        help="Rolling windows to use (default: auto-selected based on frequency)"
    )
    
    # Model input arguments
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Length of input sequences (default: 30)"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=5,
        help="Number of days to forecast (default: 5)"
    )
    parser.add_argument(
        "--target-column",
        default="close_stock_price",
        help="Target column for prediction (default: close_stock_price)"
    )
    
    # General arguments
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh data (ignore cache)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR / 'model_ready'),
        help="Output directory for model-ready data"
    )
    
    return parser.parse_args()

def run_feature_pipeline(args):
    """
    Run the feature pipeline with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Starting feature pipeline for {args.symbol}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data integrator
    integrator = DataIntegrator()
    
    # Step 1: Collect and register data
    logger.info("Step 1: Collecting and registering data")
    
    # Collect stock data
    stock_data = integrator.collect_and_register_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        force_refresh=args.force_refresh
    )
    
    if stock_data is None or stock_data.empty:
        logger.error("Failed to collect stock data, aborting pipeline")
        return
    
    logger.info(f"Collected stock data with {len(stock_data)} rows")
    
    # Collect macro data
    macro_data = integrator.collect_and_register_macro_data(
        sources=args.macro_sources,
        force_refresh=args.force_refresh
    )
    
    macro_status = {k: "Success" if v is not None and not v.empty else "Failed" 
                   for k, v in macro_data.items()}
    logger.info(f"Macro data collection status: {macro_status}")
    
    # Step 2: Create aligned features
    logger.info("Step 2: Creating aligned features")
    
    # Determine feature ID for the aligned dataset
    base_feature_id = f"{args.symbol.replace('.', '_')}_with_macro"
    
    # Create aligned features
    aligned_data = integrator.create_aligned_features(
        feature_id=base_feature_id,
        symbol=args.symbol,
        macro_sources=args.macro_sources,
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        force_refresh=args.force_refresh
    )
    
    if aligned_data is None or aligned_data.empty:
        logger.error("Failed to create aligned features, aborting pipeline")
        return
    
    logger.info(f"Created aligned feature set with {len(aligned_data)} rows and {len(aligned_data.columns)} columns")
    
    # Step 3: Generate advanced features
    logger.info("Step 3: Generating advanced features")
    
    advanced_features = integrator.generate_advanced_features(
        base_feature_id=base_feature_id,
        include_time_features=args.include_time,
        include_lag_features=args.include_lag,
        include_rolling_features=args.include_rolling,
        lag_periods=args.lag_periods,
        rolling_windows=args.rolling_windows
    )
    
    if not advanced_features:
        logger.warning("No advanced features generated")
    else:
        for feature_type, data in advanced_features.items():
            logger.info(f"Generated {feature_type} features with {len(data)} rows and {len(data.columns)} columns")
    
    # Step 4: Create model inputs
    logger.info("Step 4: Creating model inputs")
    
    # Determine feature IDs to include
    feature_ids = [base_feature_id]
    if advanced_features:
        # Add derived feature IDs
        if args.include_time:
            feature_ids.append(f"{base_feature_id}_time")
        if args.include_lag:
            feature_ids.append(f"{base_feature_id}_lag")
        if args.include_rolling:
            feature_ids.append(f"{base_feature_id}_rolling")
    
    # Create model inputs
    model_inputs = integrator.create_model_inputs(
        feature_ids=feature_ids,
        target_column=args.target_column,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon
    )
    
    if not model_inputs:
        logger.error("Failed to create model inputs, aborting pipeline")
        return
    
    # Step 5: Save model-ready data
    logger.info("Step 5: Saving model-ready data")
    
    # Save train, validation, and test sets
    try:
        np.save(output_dir / 'x_train.npy', model_inputs['x_train'])
        np.save(output_dir / 'y_train.npy', model_inputs['y_train'])
        np.save(output_dir / 'x_val.npy', model_inputs['x_val'])
        np.save(output_dir / 'y_val.npy', model_inputs['y_val'])
        np.save(output_dir / 'x_test.npy', model_inputs['x_test'])
        np.save(output_dir / 'y_test.npy', model_inputs['y_test'])
        
        # Save metadata
        metadata = {
            'feature_columns': model_inputs['feature_columns'],
            'target_column': model_inputs['target_column'],
            'sequence_length': model_inputs['sequence_length'],
            'forecast_horizon': model_inputs['forecast_horizon'],
            'train_samples': len(model_inputs['y_train']),
            'val_samples': len(model_inputs['y_val']),
            'test_samples': len(model_inputs['y_test']),
            'feature_dimension': model_inputs['x_train'].shape[2],
            'train_dates': [str(d) for d in model_inputs['train_dates']],
            'val_dates': [str(d) for d in model_inputs['val_dates']],
            'test_dates': [str(d) for d in model_inputs['test_dates']],
            'scaler_info': {
                k: {'mean': float(v['mean']), 'std': float(v['std'])}
                for k, v in model_inputs['scaler_dict'].items()
            } if model_inputs.get('scaler_dict') else {}
        }
        
        # Save as JSON
        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model-ready data to {output_dir}")
        logger.info(f"Train: {model_inputs['x_train'].shape}, Val: {model_inputs['x_val'].shape}, Test: {model_inputs['x_test'].shape}")
        
    except Exception as e:
        logger.error(f"Error saving model-ready data: {e}", exc_info=True)
        return
    
    logger.info("Feature pipeline completed successfully")

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Ensure at least one advanced feature type is selected
    if not any([args.include_time, args.include_lag, args.include_rolling]):
        args.include_time = True
        args.include_lag = True
        args.include_rolling = True
        logger.info("No specific feature types selected, enabling all advanced features")
    
    # Run the pipeline
    run_feature_pipeline(args)

if __name__ == "__main__":
    main() 