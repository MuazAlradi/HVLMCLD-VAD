#!/usr/bin/env python3
"""
HVLMCLD-VAD Pipeline Setup Script
Creates the default directory structure and configuration files for the pipeline.
"""

import argparse
import os
from pathlib import Path


def create_directory_structure(data_dir, save_dir):
    """Create the required directory structure for the HVLMCLD-VAD pipeline."""
    
    data_path = Path(data_dir)
    save_path = Path(save_dir)
    
    # Create main directories
    data_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # UCF Crime dataset structure
    ucf_base = data_path / "ucf_crime"
    ucf_videos = ucf_base / "videos"
    ucf_labels = ucf_base / "labels"
    
    # UCF Crime video categories
    ucf_categories = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", 
        "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", 
        "Stealing", "Vandalism", "Normal"
    ]
    
    for category in ucf_categories:
        (ucf_videos / category).mkdir(parents=True, exist_ok=True)
    
    ucf_labels.mkdir(parents=True, exist_ok=True)
    
    # Shanghai Tech dataset structure
    sht_base = data_path / "shanghai_tech"
    sht_videos = sht_base / "videos"
    sht_labels = sht_base / "labels"
    
    sht_videos.mkdir(parents=True, exist_ok=True)
    sht_labels.mkdir(parents=True, exist_ok=True)
    
    # XD Violence dataset structure
    xdv_base = data_path / "xd_violence"
    xdv_videos = xdv_base / "videos"
    xdv_labels = xdv_base / "labels"
    
    xdv_videos.mkdir(parents=True, exist_ok=True)
    xdv_labels.mkdir(parents=True, exist_ok=True)
    
    # Keyword dictionaries
    keyword_dir = data_path / "keyword_dictionaries"
    keyword_dir.mkdir(parents=True, exist_ok=True)
    
    # GPT descriptions
    gpt_dir = data_path / "gpt_descriptions"
    gpt_dir.mkdir(parents=True, exist_ok=True)


def create_config_file(workspace_dir, data_dir, save_dir):
    """Create a configuration file with the paths."""
    
    config_content = f"""# HVLMCLD-VAD Pipeline Configuration
# Generated automatically by create_default_local_file.py

WORKSPACE_DIR = "{os.path.abspath(workspace_dir)}"
DATA_DIR = "{os.path.abspath(data_dir)}"
SAVE_DIR = "{os.path.abspath(save_dir)}"

# Dataset paths
UCF_CRIME_PATH = "{os.path.abspath(data_dir)}/ucf_crime"
SHANGHAI_TECH_PATH = "{os.path.abspath(data_dir)}/shanghai_tech"
XD_VIOLENCE_PATH = "{os.path.abspath(data_dir)}/xd_violence"

# Keyword and description paths
KEYWORD_DICT_PATH = "{os.path.abspath(data_dir)}/keyword_dictionaries"
GPT_DESCRIPTIONS_PATH = "{os.path.abspath(data_dir)}/gpt_descriptions"
"""
    
    config_path = Path(workspace_dir) / "config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created configuration file: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Set up HVLMCLD-VAD pipeline directory structure and configuration"
    )
    parser.add_argument(
        "--workspace_dir", 
        type=str, 
        default=".", 
        help="Workspace directory (default: current directory)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data", 
        help="Data directory path (default: ./data)"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./output", 
        help="Output/save directory path (default: ./output)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up HVLMCLD-VAD Pipeline...")
    print(f"Workspace: {os.path.abspath(args.workspace_dir)}")
    print(f"Data directory: {os.path.abspath(args.data_dir)}")
    print(f"Save directory: {os.path.abspath(args.save_dir)}")
    print()
    
    # Create directory structure
    print("📁 Creating directory structure...")
    create_directory_structure(args.data_dir, args.save_dir)
    print("✅ Directory structure created!")
    print()
    
    # Create configuration file
    print("⚙️ Creating configuration file...")
    create_config_file(args.workspace_dir, args.data_dir, args.save_dir)
    print("✅ Configuration file created!")
    print()
    
    print("🎉 HVLMCLD-VAD Pipeline setup complete!")
    print()
    print("📋 Next steps:")
    print("1. Place your video datasets in the respective video directories")
    print("2. Add your label files to the labels directories")
    print("3. Add keyword dictionaries and GPT descriptions as needed")
    print("4. Import the config.py file in your pipeline scripts")


if __name__ == "__main__":
    main()
