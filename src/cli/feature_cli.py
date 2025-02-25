#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Pipeline Script

This script runs the feature generation pipeline for stock prediction model.
It handles:
1. Data collection (stock and macro)
2. Feature generation
3. Model input preparation
"""

import argparse
import logging
import logging.config
import numpy as np
import json
from pathlib import Path

# Import from project
try:
    from src.core.config import (
        LOGGING_CONFIG, DEFAULT_TICKER_BS, START_DATE, END_DATE, DATA_DIR
    )
    from src.features.integration import DataIntegrator
except ImportError:
    from ..core.config import (
        LOGGING_CONFIG, DEFAULT_TICKER_BS, START_DATE, END_DATE, DATA_DIR
    )
    from ..features.integration import DataIntegrator

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature Pipeline for Stock Prediction")
    
    # Essential arguments only
    parser.add_argument("--symbol", default=DEFAULT_TICKER_BS, 
                        help=f"Stock symbol (default: {DEFAULT_TICKER_BS})")
    parser.add_argument("--start-date", default=START_DATE,
                        help=f"Start date (default: {START_DATE})")
    parser.add_argument("--end-date", default=END_DATE,
                        help=f"End date (default: {END_DATE})")
    parser.add_argument("--frequency", choices=['daily', 'weekly', 'monthly'], default='daily',
                        help="Target frequency (default: daily)")
    parser.add_argument("--output-dir", default=str(DATA_DIR / 'model_ready'),
                        help="Output directory (default: data/model_ready)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force refresh data")
    
    # Feature types (simplified)
    parser.add_argument("--feature-types", nargs="+", choices=['time', 'lag', 'rolling'],
                        default=['time', 'lag', 'rolling'],
                        help="Feature types to include (default: all)")
    
    # Model parameters (simplified)
    parser.add_argument("--sequence-length", type=int, default=30,
                        help="Sequence length (default: 30)")
    parser.add_argument("--forecast-horizon", type=int, default=5,
                        help="Forecast horizon (default: 5)")
    
    args = parser.parse_args()
    
    # Process feature types
    args.include_time = 'time' in args.feature_types
    args.include_lag = 'lag' in args.feature_types
    args.include_rolling = 'rolling' in args.feature_types
    
    return args

def run_feature_pipeline(args):
    """Run the feature pipeline with the specified arguments."""
    logger.info(f"Starting feature pipeline for {args.symbol}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data integrator
    integrator = DataIntegrator()
    
    # Step 1: Collect data
    logger.info("Collecting stock data")
    stock_data = integrator.collect_and_register_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        force_refresh=args.force_refresh
    )
    
    if stock_data is None or stock_data.empty:
        logger.error("Failed to collect stock data")
        return
    
    logger.info("Collecting macro data")
    macro_data = integrator.collect_and_register_macro_data(
        force_refresh=args.force_refresh
    )
    
    # Step 2: Create aligned features
    logger.info("Creating aligned features")
    base_feature_id = f"{args.symbol.replace('.', '_')}_with_macro"
    
    # Check if any macro data was collected
    has_macro_data = bool(macro_data and any(not getattr(v, 'empty', True) for v in macro_data.values()))
    
    try:
        if has_macro_data:
            aligned_data = integrator.create_aligned_features(
                feature_id=base_feature_id,
                symbol=args.symbol,
                frequency=args.frequency,
                force_refresh=args.force_refresh
            )
        else:
            logger.warning("No macro data available. Creating stock-only features.")
            base_feature_id = f"{args.symbol.replace('.', '_')}_only"
            aligned_data = integrator.create_aligned_features(
                feature_id=base_feature_id,
                symbol=args.symbol,
                macro_sources=None,
                frequency=args.frequency,
                force_refresh=args.force_refresh
            )
    except Exception as e:
        logger.error(f"Failed to create features: {e}")
        return
    
    if aligned_data is None or aligned_data.empty:
        logger.error("Failed to create aligned features")
        return
    
    # Step 3: Generate advanced features
    logger.info("Generating advanced features")
    advanced_features = integrator.generate_advanced_features(
        base_feature_id=base_feature_id,
        include_time_features=args.include_time,
        include_lag_features=args.include_lag,
        include_rolling_features=args.include_rolling
    )
    
    # Step 4: Create model inputs
    logger.info("Creating model inputs")
    feature_ids = [base_feature_id]
    
    # Add advanced feature IDs if they were generated
    if advanced_features:
        if args.include_time:
            feature_ids.append(f"{base_feature_id}_time")
        if args.include_lag:
            feature_ids.append(f"{base_feature_id}_lag")
        if args.include_rolling:
            feature_ids.append(f"{base_feature_id}_rolling")
    
    model_inputs = integrator.create_model_inputs(
        feature_ids=feature_ids,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon
    )
    
    if not model_inputs:
        logger.error("Failed to create model inputs")
        return
    
    # Step 5: Save model-ready data
    logger.info("Saving model-ready data")
    try:
        # Save arrays
        np.save(output_dir / 'x_train.npy', model_inputs['x_train'])
        np.save(output_dir / 'y_train.npy', model_inputs['y_train'])
        np.save(output_dir / 'x_val.npy', model_inputs['x_val'])
        np.save(output_dir / 'y_val.npy', model_inputs['y_val'])
        np.save(output_dir / 'x_test.npy', model_inputs['x_test'])
        np.save(output_dir / 'y_test.npy', model_inputs['y_test'])
        
        # Save metadata (simplified)
        metadata = {
            'feature_columns': model_inputs['feature_columns'],
            'target_column': model_inputs['target_column'],
            'sequence_length': model_inputs['sequence_length'],
            'forecast_horizon': model_inputs['forecast_horizon'],
            'samples': {
                'train': len(model_inputs['y_train']),
                'val': len(model_inputs['y_val']),
                'test': len(model_inputs['y_test']),
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model-ready data to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving model-ready data: {e}")
        return
    
    logger.info("Feature pipeline completed successfully")

def main():
    """Main entry point."""
    args = parse_arguments()
    run_feature_pipeline(args)

if __name__ == "__main__":
    main() 