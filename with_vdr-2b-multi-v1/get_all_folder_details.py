# Standard library imports
import os

def get_subfolder_paths(main_folder):
    """
    Retrieve paths of all subfolders within the main folder.
    
    This function is specifically designed to work with the output of the PDF to image
    conversion process, where each PDF gets its own subfolder of images.
    
    Args:
        main_folder (str): Path to the main folder containing converted image subfolders
        
    Returns:
        list: List of absolute paths to all subfolders
        
    Example:
        >>> main_folder = "/path/to/converted/images"
        >>> subfolder_paths = get_subfolder_paths(main_folder)
        >>> print(subfolder_paths)
        ['/path/to/converted/images/pdf1', '/path/to/converted/images/pdf2']
    """
    subfolder_paths = []
    
    # Iterate through items in the main folder
    for item in os.listdir(main_folder):
        # Construct full path
        full_path = os.path.join(main_folder, item)
        # Check if the item is a directory
        if os.path.isdir(full_path):
            # Add to list using raw string format for Windows compatibility
            subfolder_paths.append(rf"{main_folder}/{item}")
            
    return subfolder_paths

# Example usage with proper path handling
if __name__ == "__main__":
    # Replace with your actual path to the converted images
    main_folder = "main folder path"
    
    # Get all subfolder paths
    subfolder_paths = get_subfolder_paths(main_folder)
    
    # Print results
    print("Found the following subfolders:")
    for path in subfolder_paths:
        print(f"- {path}")