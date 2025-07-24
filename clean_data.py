#!/usr/bin/env python3
"""
Clean Data Script

This script deletes all data files generated during the AI Go analysis pipeline.
It removes numpy arrays, JSON files, SGF files, NPZ files, and other generated data
while preserving source code and configuration files.

Usage:
    python clean_data.py
"""

import os
import shutil
import glob
from pathlib import Path


def delete_data_files():
    """Delete all data files in the project."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    print("🧹 Cleaning data files from AI Go analysis project...")
    print(f"Project root: {project_root}")
    print()
    
    # Define data file patterns to delete
    data_patterns = [
        # Numpy arrays
        "*.npy",
        "*.npz",
        
        # JSON data files (but not config files)
        "analysis_*.json",
        "pooled_meta.json",
        "nmf_meta.json",
        "strong_positions.json",
        
        # Text data files
        "pos_index_to_npz.txt",
        "strong_positions_summary.csv",
        
        # SGF files
        "*.sgf",
        "*.sgfs",
        
        # Log files
        "*.log",
        
        # Generated HTML reports
        "html_reports/*.html",
        
        # Part files
        "part*_rank*_pos*.npy",
    ]
    
    # Directories to clean (but not delete the directory itself)
    data_directories = [
        "3_extract_activations/activations",
        "5_inspect_parts/html_reports",
    ]
    
    # Directories to completely remove
    remove_directories = [
        "selfplay_out",
    ]
    
    total_deleted = 0
    
    # Delete files matching patterns
    for pattern in data_patterns:
        files = list(project_root.rglob(pattern))
        for file_path in files:
            if file_path.is_file():
                try:
                    file_path.unlink()
                    print(f"🗑️  Deleted: {file_path.relative_to(project_root)}")
                    total_deleted += 1
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
    
    # Clean data directories (remove contents but keep directory)
    for dir_path in data_directories:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            try:
                for item in full_path.iterdir():
                    if item.is_file():
                        item.unlink()
                        print(f"🗑️  Deleted: {item.relative_to(project_root)}")
                        total_deleted += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"🗑️  Deleted directory: {item.relative_to(project_root)}")
                        total_deleted += 1
            except Exception as e:
                print(f"❌ Error cleaning directory {full_path}: {e}")
    
    # Remove entire directories
    for dir_path in remove_directories:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            try:
                shutil.rmtree(full_path)
                print(f"🗑️  Removed directory: {dir_path}")
                total_deleted += 1
            except Exception as e:
                print(f"❌ Error removing directory {full_path}: {e}")
    
    print()
    print(f"✅ Cleanup complete! Deleted {total_deleted} files/directories.")
    print()
    print("Preserved files:")
    print("- Source code (.py files)")
    print("- Configuration files (.cfg, .yml)")
    print("- Documentation (.md files)")
    print("- Concept map (concept_map.json)")
    print("- License and README files")


if __name__ == "__main__":
    # Ask for confirmation before proceeding
    print("⚠️  WARNING: This will delete all generated data files!")
    print("This includes:")
    print("- Numpy arrays (.npy, .npz)")
    print("- Analysis JSON files")
    print("- SGF game files")
    print("- HTML reports")
    print("- Selfplay output data")
    print("- Log files")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        delete_data_files()
    else:
        print("❌ Cleanup cancelled.") 