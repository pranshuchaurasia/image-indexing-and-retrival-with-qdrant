"""
PDF to Image Converter

Converts all PDF files in a folder to PNG images.
Each PDF page becomes a separate image file.

Usage:
    python convert_pdf_to_image.py

Requirements:
    - pdf2image: pip install pdf2image
    - Poppler: Download from https://github.com/oschwartz10612/poppler-windows/releases
    - Set poppler_path in .env file
"""

import os
from pathlib import Path
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get poppler path from environment
POPPLER_PATH = os.getenv('poppler_path')
if not POPPLER_PATH:
    print("WARNING: poppler_path not set in .env file")
    print("PDF conversion may fail. Please set poppler_path in your .env file")


def convert_pdf_to_images(pdf_folder: str, output_folder: str = None, thread_count: int = 6):
    """
    Convert all PDFs in the specified folder to images.
    Each page of each PDF will be saved as a separate PNG file.
    
    Args:
        pdf_folder (str): Path to the folder containing PDF files
        output_folder (str): Optional output folder path. Defaults to pdf_folder/new_converted_images
        thread_count (int): Number of threads for parallel processing
        
    Directory Structure Created:
        output_folder/
        ├── pdf1_name/
        │   ├── pdf1_name_page_1.png
        │   ├── pdf1_name_page_2.png
        │   └── ...
        ├── pdf2_name/
        │   └── ...
        └── ...
    """
    # Create output directory if it doesn't exist
    if output_folder is None:
        output_folder = os.path.join(pdf_folder, 'new_converted_images')
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert")
    
    for pdf_file in pdf_files:
        try:
            # Get the PDF file name without extension
            pdf_name = Path(pdf_file).stem
            pdf_path = os.path.join(pdf_folder, pdf_file)
            saved_img_path = os.path.join(output_folder, pdf_name)
            os.makedirs(saved_img_path, exist_ok=True)
           
            print(f"Converting {pdf_file}...")
            
            # Convert PDF to images
            if POPPLER_PATH:
                images = convert_from_path(
                    pdf_path, 
                    thread_count=thread_count, 
                    poppler_path=POPPLER_PATH
                )
            else:
                # Try without poppler_path (works if poppler is in system PATH)
                images = convert_from_path(pdf_path, thread_count=thread_count)
            
            # Save each page as an image
            for i, image in enumerate(images):
                output_file = f"{pdf_name}_page_{i+1}.png"
                output_path = os.path.join(saved_img_path, output_file)
                image.save(output_path, 'PNG')
                print(f"  Saved {output_file}")
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    print("\nConversion completed!")


if __name__ == "__main__":
    # Get folder path from environment or use default
    pdf_folder = os.getenv('pdf_input_folder', r"D:\Pranshu\Data")
    
    print(f"PDF Input Folder: {pdf_folder}")
    print(f"Poppler Path: {POPPLER_PATH}")
    
    convert_pdf_to_images(pdf_folder)
