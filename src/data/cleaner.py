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

def clean_directory(directory: Path, keep_gitkeep: bool = True, dry_run: bool = False, recursive: bool = False) -> int:
    """
    Clean all files in a directory while preserving the directory structure.
    
    Args:
        directory: The directory to clean
        keep_gitkeep: Whether to preserve .gitkeep files
        dry_run: If True, only simulate deletion
        recursive: If True, also clean subdirectories
        
    Returns:
        Number of files removed
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0
        
    count = 0
    
    # First clean files in the current directory
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
    
    # If recursive, also clean all subdirectories
    if recursive:
        for subdir in directory.glob('*/'):
            if subdir.is_dir():
                # Skip .git directories
                if '.git' in str(subdir):
                    continue
                    
                logger.info(f"Cleaning subdirectory: {subdir}")
                count += clean_directory(subdir, keep_gitkeep, dry_run, recursive=True)
                
    return count

def clean_data_directories(dry_run: bool = False) -> dict:
    """
    Clean all data directories.
    
    Args:
        dry_run: If True, only simulate deletion
        
    Returns:
        Dictionary with counts of removed files per directory
    """
    results = {}
    
    # All directories to clean with appropriate settings
    dirs_to_clean = [
        ('raw', RAW_DATA_DIR, False),         # (name, path, recursive)
        ('processed', PROCESSED_DATA_DIR, False),
        ('cache', CACHE_DIR, False),
        ('macro', MACRO_DATA_DIR, False),
        ('feature_store', FEATURE_STORE_DIR, True),  # Use recursive for feature_store
        ('model_ready', MODEL_READY_DIR, False)
    ]
        
    # Clean each directory
    for name, directory, recursive in dirs_to_clean:
        logger.info(f"Cleaning directory: {name}")
        count = clean_directory(directory, keep_gitkeep=True, dry_run=dry_run, recursive=recursive)
        results[name] = count
        
    return results

def main():
    """Main entry point for the data cleaning utility."""
    parser = argparse.ArgumentParser(
        description="Clean all data directories while preserving directory structure."
    )
    
    # Keep only the dry-run option for safety
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Simulate cleaning without actually removing files"
    )
    
    args = parser.parse_args()
    
    # Get confirmation unless it's a dry run
    if not args.dry_run:
        confirmation = input(f"Are you sure you want to clean ALL data directories? [y/N]: ")
        if confirmation.lower() not in ['y', 'yes']:
            logger.info("Operation cancelled by user.")
            return
    
    # Perform cleaning of ALL directories
    operation = "Simulating" if args.dry_run else "Cleaning"
    logger.info(f"{operation} all data directories...")
    
    results = clean_data_directories(dry_run=args.dry_run)
    
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