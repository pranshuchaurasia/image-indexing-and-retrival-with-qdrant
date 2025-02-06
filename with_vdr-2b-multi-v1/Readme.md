# Visual Document Retrieval System

An implementation of a visual document retrieval system using the VDR-2B-Multi-V1 model, which enables multilingual document search through image embeddings without requiring OCR.

## Features

- PDF to Image Conversion
- Multi-folder Processing
- Incremental Vector Database Indexing
- Multi-lingual Visual Document Search
- Cross-lingual Retrieval Support
- Support for Multiple Document Formats


## Project Structure

```
project/
├── .env                    # Environment variables
├── convert_pdf_to_image.py # PDF to image conversion
├── get_all_folder_details.py # Folder path retrieval
├── incremental_indexing_vdr_2b_multi_v1.py # Vector indexing
└── query_vdr_2b_multi_v1.py # Search implementation
```

## Setup Instructions

1. Install required dependencies:
```bash
pip install pdf2image qdrant-client llama-index pillow python-dotenv tqdm
```

2. Install Poppler for PDF processing:


3. Create a `.env` file with the following variables:
```
vdr_2b_multi_v1=Xenova/vdr-2b-multi-v1
collection_name_vdr=your_collection_name
```

## Usage Flow

1. **Convert PDFs to Images**
   ```python
   python convert_pdf_to_image.py
   ```
   - Input: Folder containing PDF files
   - Output: Separate folders for each PDF containing page images

2. **Get Folder Paths**
   ```python
   python get_all_folder_details.py
   ```
   - Retrieves paths of all converted image folders

3. **Index Images**
   ```python
   python incremental_indexing_vdr_2b_multi_v1.py
   ```
   - Creates/updates vector database with image embeddings
   - Supports incremental updates

4. **Search Documents**
   ```python
   python query_vdr_2b_multi_v1.py
   ```
   - Performs visual similarity search
   - Returns relevant document pages

## Model Details: VDR-2B-Multi-V1

The VDR-2B-Multi-V1 model is designed for multilingual visual document retrieval:

- Supports Italian, Spanish, English, French, and German
- Enables cross-lingual document search
- Uses Matryoshka Representation Learning
- Processes document screenshots without OCR
- Generates 1536-dimensional embeddings

## Note:
I tried it for an offline run. Feel free to reach out if you have any queries.