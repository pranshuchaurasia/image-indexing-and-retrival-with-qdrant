"""
Folder Scanner Utility

Retrieves paths of all subfolders within a main folder.
Useful for getting list of converted image folders for indexing.

Usage:
    python get_all_folder_details.py
    
    Or import and use:
    from shared.get_all_folder_details import get_subfolder_paths
    paths = get_subfolder_paths("/path/to/folder")
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_subfolder_paths(main_folder: str) -> list:
    """
    Retrieve paths of all subfolders within the main folder.
    
    This function is designed to work with the output of the PDF to image
    conversion process, where each PDF gets its own subfolder of images.
    
    Args:
        main_folder (str): Path to the main folder containing subfolders
        
    Returns:
        list: List of absolute paths to all subfolders
        
    Example:
        >>> paths = get_subfolder_paths("/path/to/converted/images")
        >>> print(paths)
        ['/path/to/converted/images/pdf1', '/path/to/converted/images/pdf2']
    """
    subfolder_paths = []
    
    if not os.path.exists(main_folder):
        print(f"Warning: Folder does not exist: {main_folder}")
        return subfolder_paths
    
    # Iterate through items in the main folder
    for item in os.listdir(main_folder):
        full_path = os.path.join(main_folder, item)
        if os.path.isdir(full_path):
            # Use forward slashes for cross-platform compatibility
            subfolder_paths.append(str(Path(full_path)))
            
    return subfolder_paths


def get_image_files(folder: str, extensions: set = None) -> list:
    """
    Get all image files in a folder and its subfolders.
    
    Args:
        folder (str): Path to search
        extensions (set): File extensions to include. Defaults to {'.jpg', '.jpeg', '.png'}
        
    Returns:
        list: List of image file paths
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png'}
    
    image_paths = []
    folder_path = Path(folder)
    
    for file_path in folder_path.rglob('*'):
        if file_path.suffix.lower() in extensions:
            image_paths.append(str(file_path))
    
    return image_paths


if __name__ == "__main__":
    # Get folder path from environment or use default
    main_folder = os.getenv('converted_images_folder', r"D:\Pranshu\Data\new_converted_images")
    
    print(f"Scanning folder: {main_folder}")
    
    subfolder_paths = get_subfolder_paths(main_folder)
    
    print(f"\nFound {len(subfolder_paths)} subfolders:")
    for path in subfolder_paths:
        print(f"  - {path}")
    
    # Optionally print image count
    total_images = 0
    for folder in subfolder_paths:
        images = get_image_files(folder)
        total_images += len(images)
    
    print(f"\nTotal images across all folders: {total_images}")
