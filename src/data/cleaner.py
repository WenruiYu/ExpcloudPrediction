# src/data/cleaner.py

import os
import argparse
import shutil
import logging
import logging.config
from pathlib import Path

# Try to import using relative imports if running as a module
try:
    from src.core.config import (
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        CACHE_DIR, MACRO_DATA_DIR, MODEL_READY_DIR, LOGGING_CONFIG
    )
    from src.features.store import FEATURE_STORE_DIR
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import (
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        CACHE_DIR, MACRO_DATA_DIR, MODEL_READY_DIR, LOGGING_CONFIG
    )
    from ..features.store import FEATURE_STORE_DIR

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def clean_directory(directory: Path, keep_gitkeep: bool = True, dry_run: bool = False) -> int:
    """
    Clean all files in a directory while preserving the directory structure.
    
    Args:
        directory: The directory to clean
        keep_gitkeep: Whether to preserve .gitkeep files
        dry_run: If True, only simulate deletion
        
    Returns:
        Number of files removed
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0
        
    count = 0
    for item in directory.glob('*'):
        if item.is_file():
            # Skip .gitkeep files if requested
            if keep_gitkeep and item.name == '.gitkeep':
                continue
                
            if dry_run:
                logger.info(f"[DRY RUN] Would remove: {item}")
            else:
                item.unlink()
                logger.info(f"Removed: {item}")
            count += 1
                
    return count

def clean_data_directories(
    raw: bool = False,
    processed: bool = False,
    cache: bool = False,
    macro: bool = False,
    feature_store: bool = False,
    model_ready: bool = False,
    all_dirs: bool = False,
    keep_gitkeep: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Clean specified data directories.
    
    Args:
        raw: Clean raw data directory
        processed: Clean processed data directory
        cache: Clean cache directory
        macro: Clean macro data directory
        feature_store: Clean feature store directory
        model_ready: Clean model-ready data directory
        all_dirs: Clean all data directories
        keep_gitkeep: Whether to preserve .gitkeep files
        dry_run: If True, only simulate deletion
        
    Returns:
        Dictionary with counts of removed files per directory
    """
    results = {}
    
    # Determine which directories to clean
    dirs_to_clean = []
    
    if all_dirs or raw:
        dirs_to_clean.append(('raw', RAW_DATA_DIR))
    if all_dirs or processed:
        dirs_to_clean.append(('processed', PROCESSED_DATA_DIR))
    if all_dirs or cache:
        dirs_to_clean.append(('cache', CACHE_DIR))
    if all_dirs or macro:
        dirs_to_clean.append(('macro', MACRO_DATA_DIR))
    if all_dirs or feature_store:
        dirs_to_clean.append(('feature_store', FEATURE_STORE_DIR))
    if all_dirs or model_ready:
        dirs_to_clean.append(('model_ready', MODEL_READY_DIR))
        
    if not dirs_to_clean:
        logger.warning("No directories selected for cleaning. Use --all or specify directories.")
        return results
        
    # Clean each directory
    for name, directory in dirs_to_clean:
        count = clean_directory(directory, keep_gitkeep, dry_run)
        results[name] = count
        
    return results

def main():
    """Main entry point for the data cleaning utility."""
    parser = argparse.ArgumentParser(
        description="Clean data files while preserving directory structure."
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Clean raw data directory"
    )
    parser.add_argument(
        '--processed',
        action='store_true',
        help="Clean processed data directory"
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help="Clean cache directory"
    )
    parser.add_argument(
        '--macro',
        action='store_true',
        help="Clean macro data directory"
    )
    parser.add_argument(
        '--feature-store',
        action='store_true',
        help="Clean feature store directory"
    )
    parser.add_argument(
        '--model-ready',
        action='store_true',
        help="Clean model-ready data directory"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Clean all data directories"
    )
    parser.add_argument(
        '--no-gitkeep',
        action='store_true',
        help="Also remove .gitkeep files"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Simulate cleaning without actually removing files"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Get confirmation unless forced
    if not args.force and not args.dry_run:
        dirs = []
        if args.all:
            dirs.append("ALL data directories")
        else:
            if args.raw:
                dirs.append("raw")
            if args.processed:
                dirs.append("processed")
            if args.cache:
                dirs.append("cache")
            if args.macro:
                dirs.append("macro")
            if args.feature_store:
                dirs.append("feature_store")
            if args.model_ready:
                dirs.append("model_ready")
                
        if not dirs:
            logger.warning("No directories selected for cleaning. Use --all or specify directories.")
            return
            
        confirmation = input(f"Are you sure you want to clean {', '.join(dirs)}? [y/N]: ")
        if confirmation.lower() not in ['y', 'yes']:
            logger.info("Operation cancelled by user.")
            return
    
    # Perform cleaning
    operation = "Simulating" if args.dry_run else "Cleaning"
    logger.info(f"{operation} data directories...")
    
    results = clean_data_directories(
        raw=args.raw,
        processed=args.processed,
        cache=args.cache,
        macro=args.macro,
        feature_store=args.feature_store,
        model_ready=args.model_ready,
        all_dirs=args.all,
        keep_gitkeep=not args.no_gitkeep,
        dry_run=args.dry_run
    )
    
    # Print summary
    if results:
        logger.info("Clean operation summary:")
        total = 0
        for directory, count in results.items():
            logger.info(f"  {directory}: {count} files removed")
            total += count
        logger.info(f"Total: {total} files {('would be ' if args.dry_run else '')}removed")
    else:
        logger.warning("No files were removed.")

if __name__ == "__main__":
    main() 