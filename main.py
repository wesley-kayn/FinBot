#!/usr/bin/env python
"""
Finbot LLM - Main Entry Point

This script initializes and runs the Finbot LLM application.
"""

import os
import sys
import shutil
from app.app import main

if __name__ == '__main__':
    # Copy the sample data files to the data directory if they're not there yet
    app_data_dir = os.path.join('app', 'data')
    os.makedirs(app_data_dir, exist_ok=True)
    
    # Check for sample files in the current directory
    sample_files = ['funds_transfer_app_features_app.json', 'NUST Bank-Product-Knowledge.xlsx']
    
    for sample_file in sample_files:
        if os.path.exists(sample_file) and not os.path.exists(os.path.join(app_data_dir, sample_file)):
            try:
                shutil.copy(sample_file, os.path.join(app_data_dir, sample_file))
                print(f"Copied {sample_file} to {app_data_dir}")
            except Exception as e:
                print(f"Error copying {sample_file}: {e}")
    
    # Run the application
    main() 