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
    
    # Define specific data files to delete (be very specific to avoid deleting code)
    specific_files_to_delete = [
        # Large data files
        "3_extract_activations/activations/pooled_rconv14.out.npy",
        "3_extract_activations/activations/pooled_meta.json",
        "3_extract_activations/activations/pos_index_to_npz.txt",
        
        # NMF analysis data
        "4_nmf_parts/nmf_activations.npy",
        "4_nmf_parts/nmf_components.npy",
        "4_nmf_parts/nmf_meta.json",
        "4_nmf_parts/3x3_validation_report.json",
        "4_nmf_parts/alpha_h_recommendations_3x3.json",
        "4_nmf_parts/alpha_h_analysis_results.json",
        
        # Analysis output data
        "5_inspect_parts/strong_positions_summary.csv",
    ]
    
    # Define data file patterns to delete (be more specific)
    data_patterns = [
        # Large numpy arrays (but not small ones that might be code)
        "**/pooled_*.npy",
        "**/nmf_*.npy",
        "**/board.npy",
        
        # Specific JSON data files (not all JSON)
        "**/analysis_*.json",
        "**/pooled_meta.json",
        "**/nmf_meta.json",
        "**/strong_positions.json",
        "**/3x3_validation_report.json",
        "**/alpha_h_recommendations_*.json",
        "**/alpha_h_analysis_results.json",
        
        # Large text data files
        "**/pos_index_to_npz.txt",
        "**/strong_positions_summary.csv",
        "**/rank_analysis_report.txt",
        
        # SGF files
        "**/*.sgf",
        "**/*.sgfs",
        
        # Log files
        "**/*.log",
        
        # Generated HTML reports
        "**/html_reports/*.html",
        
        # Part files
        "**/part*_rank*_pos*.npy",
    ]
    
    # Directories to clean (but not delete the directory itself)
    data_directories = [
        "3_extract_activations/activations",
        "5_inspect_parts/html_reports",
        "4_nmf_parts/rank_analysis",
    ]
    
    # Directories to completely remove
    remove_directories = [
        "selfplay_out",
        "5_inspect_parts/output",
    ]
    
    total_deleted = 0
    
    # Delete specific files first
    for file_path in specific_files_to_delete:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            try:
                full_path.unlink()
                print(f"🗑️  Deleted: {file_path}")
                total_deleted += 1
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
    
    # Delete files matching patterns
    for pattern in data_patterns:
        files = list(project_root.rglob(pattern))
        for file_path in files:
            # Skip files in models and KataGo directories
            if "models" in file_path.parts or "KataGo" in file_path.parts:
                continue
            # Skip README files and documentation
            if file_path.name in ["README.md", "QUICK_REFERENCE.md"]:
                continue
            # Skip small files that might be code
            if file_path.stat().st_size < 10000:  # Skip files smaller than 10KB
                continue
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
                        # Skip README and documentation files
                        if item.name in ["README.md", "QUICK_REFERENCE.md"]:
                            continue
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
    print("- KataGo source code")
    print("- Model files")


if __name__ == "__main__":
    # Ask for confirmation before proceeding
    print("⚠️  WARNING: This will delete all generated data files!")
    print("This includes:")
    print("- Large numpy arrays (.npy, .npz)")
    print("- Analysis JSON files")
    print("- SGF game files")
    print("- HTML reports")
    print("- Selfplay output data")
    print("- Log files")
    print("- NMF analysis files and components")
    print("- Rank analysis files")
    print("- Output analysis directories")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        delete_data_files()
    else:
        print("❌ Cleanup cancelled.") 