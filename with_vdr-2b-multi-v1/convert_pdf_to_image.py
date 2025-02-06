from pdf2image import convert_from_path
import os
from pathlib import Path

def convert_pdf_to_images(pdf_folder):
    """
    Convert all PDFs in the specified folder to images.
    Each page of each PDF will be saved as a separate PNG file in a dedicated subfolder.
    
    Args:
        pdf_folder (str): Path to the folder containing PDF files
        
    Directory Structure Created:
        pdf_folder/
        └── new_converted_images/
            ├── pdf1_name/
            │   ├── pdf1_name_page_1.png
            │   ├── pdf1_name_page_2.png
            │   └── ...
            ├── pdf2_name/
            │   ├── pdf2_name_page_1.png
            │   ├── pdf2_name_page_2.png
            │   └── ...
            └── ...
    
    Requirements:
        - Poppler needs to be installed and accessible

    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(pdf_folder, 'new_converted_images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return
    
    for pdf_file in pdf_files:
        try:
            # Get the PDF file name without extension
            pdf_name = Path(pdf_file).stem
            pdf_path = os.path.join(pdf_folder, pdf_file)
            saved_img_path=os.path.join(output_dir,pdf_name)
            os.makedirs(saved_img_path, exist_ok=True)
           
            print(f"Converting {pdf_file}...")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path,thread_count=6,poppler_path="add poppler bin path")
            
            # Save each page as an image
            for i, image in enumerate(images):
                # Generate output filename: pdf_name_page_number.png
                output_file = f"{pdf_name}_page_{i+1}.png"
                output_path = os.path.join(saved_img_path, output_file)
                
                # Save the image
                image.save(output_path, 'PNG')
                print(f"Saved {output_file}")
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    print("\nConversion completed!")

# Example usage
if __name__ == "__main__":
    # Replace 'X' with your PDF folder path
    pdf_folder = pdf_folder_path
    convert_pdf_to_images(pdf_folder)