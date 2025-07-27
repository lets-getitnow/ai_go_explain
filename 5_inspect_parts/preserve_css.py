#!/usr/bin/env python3
"""
CSS Preservation Script

This script helps preserve CSS and JS files when cleaning up HTML reports.
It backs up the besogo/ directory before cleanup and restores it afterward.
"""

import os
import shutil
import tempfile
from pathlib import Path

def preserve_css_files():
    """Preserve CSS files by backing them up before cleanup."""
    html_reports_dir = Path("html_reports")
    besogo_dir = html_reports_dir / "besogo"
    
    if not besogo_dir.exists():
        print("No besogo directory found to preserve.")
        return None
    
    # Create temporary backup
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_path = Path(temp_dir) / "besogo_backup"
        shutil.copytree(besogo_dir, backup_path)
        print(f"✅ CSS files backed up to temporary location")
        return backup_path

def restore_css_files(backup_path):
    """Restore CSS files from backup after cleanup."""
    if backup_path is None:
        print("No backup to restore.")
        return
    
    html_reports_dir = Path("html_reports")
    besogo_dir = html_reports_dir / "besogo"
    
    # Restore the files
    if besogo_dir.exists():
        shutil.rmtree(besogo_dir)
    shutil.copytree(backup_path, besogo_dir)
    print(f"✅ CSS files restored from backup")

def cleanup_with_css_preservation():
    """Clean up HTML reports while preserving CSS files."""
    print("🧹 Cleaning up HTML reports while preserving CSS...")
    
    # Backup CSS files
    backup_path = preserve_css_files()
    
    # Clean up HTML files (but not besogo directory)
    html_reports_dir = Path("html_reports")
    if html_reports_dir.exists():
        for item in html_reports_dir.iterdir():
            if item.is_file() and item.suffix == '.html':
                item.unlink()
                print(f"🗑️  Deleted {item.name}")
            elif item.is_dir() and item.name != 'besogo':
                shutil.rmtree(item)
                print(f"🗑️  Deleted directory {item.name}")
    
    # Restore CSS files
    restore_css_files(backup_path)
    print("✅ Cleanup complete with CSS preserved")

if __name__ == "__main__":
    cleanup_with_css_preservation() 