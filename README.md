# Image Indexing and Retrieval with Qdrant

A visual document retrieval system using **Qdrant** vector database with four embedding models: **ColPali**, **ColQwen2**, **VDR-2B-Multi-V1**, and **Jina Embeddings v4**.

> **Note:** This code is designed to run models **offline** from local paths. No internet connection is required during inference.
> **Note:** The Jina V4 text embedding scripts (`with_jina_text_v4`) are added primarily for reference to demonstrate single-vector text indexing, as they share similarities with other dense retrieval methods.

## Repository Structure

```
├── .env-example           # Environment configuration template
├── shared/                # Common utilities
│   ├── convert_pdf_to_image.py
│   └── get_all_folder_details.py
├── with_vdr-2b-multi-v1/
├── with_colpali_v1.3/
├── with_colqwen2_v1.0/
├── with_jina_embeddings_v4/
└── with_jina_text_v4/
```

## Models

| Model | HuggingFace Link |
|-------|------------------|
| **VDR-2B-Multi-V1** | [vdr-2b-multi-v1](https://huggingface.co/llamaindex/vdr-2b-multi-v1) |
| **ColPali v1.3** | [colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) |
| **ColQwen2 v1.0** | [colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) |
| **Jina Embeddings v4** | [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) |

## Key Features

- **Efficiency**: The integration of these models allows for rapid indexing and retrieval of large datasets, significantly reducing search times.
- **Multimodal Capabilities**: By processing both images and text, these models enhance the semantic understanding of queries, leading to more relevant search results.
- **Scalability**: Qdrant's architecture supports high-performance vector similarity searches, making it suitable for applications involving large-scale image datasets.

The combination of Qdrant with ColPali, ColQwen, and VDR-2B-Multi-V1 provides a robust framework for efficient image indexing and retrieval. This approach not only improves the accuracy of search results but also enhances user experience by enabling quick access to relevant information across diverse datasets.

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU
- Qdrant running on localhost:6333
- Poppler (for PDF conversion)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd image-indexing-and-retrival-with-qdrant

# Create virtual environment with uv
uv venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA 12.8 support 
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
uv pip install -r requirements.txt
```

> **Note:** On Windows, `pillow` installation via pip sometimes has issues. 

### Configuration

```bash
# Copy environment template
cp .env-example .env

# Edit .env with your local model paths and settings

# IMPORTANT: Configure Data Paths
# 1. pdf_input_folder: Folder containing your raw PDF files
# 2. converted_images_folder: Folder where images will be saved (and read from for indexing)
```

### Usage

#### 1. Convert PDFs to Images (First Step)
```bash
python shared/convert_pdf_to_image.py
```

#### 2. Index and Search (Choose One Model)

**Option A: ColPali (Multi-Vector, 1030x128D)**
```bash
# Index
python with_colpali_v1.3/incremental_indexing_colpali_with_qdrant.py

# Search (Must use ColPali searcher)
python with_colpali_v1.3/qdrant_query_with_colpali.py
```

**Option B: ColQwen2 (Multi-Vector, Qwen-based)**
```bash
# Index
python with_colqwen2_v1.0/incremental_indexing_colqwen_with_qdrant.py

# Search (Must use ColQwen searcher)
python with_colqwen2_v1.0/qdrant_query_with_colqwen.py
```

**Option C: VDR-2B-Multi-V1 (Single Vector, 1536D)**
```bash
# Index
 python with_vdr-2b-multi-v1/incremental_indexing_vdr_2b_multi_v1.py

# Search (Must use VDR searcher)
python with_vdr-2b-multi-v1/query_vdr_2b_multi_v1.py
```

**Option D: Jina V4 (Image Multivector, 128D)**
```bash
# Index
python with_jina_embeddings_v4/jina_v4_image_indexer.py

# Search (Multivector MaxSim)
python with_jina_embeddings_v4/jina_v4_image_retrieval.py
```

**Option E: Jina V4 (Text Single-vector, 2048D)**
```bash
# Index (with sample JSON)
python with_jina_text_v4/jina_v4_text_indexer.py

# Search (Dense Vector)
python with_jina_text_v4/jina_v4_text_retrieval.py
```

## Notes

- Models are loaded offline using `local_files_only=True`
- Recommended batch size: 16 for multi-vector models (ColPali/ColQwen)
- Tested on **Windows** with **NVIDIA RTX 6000 ADA 48GB**
- Runs **without flash-attention-2** (not required)
