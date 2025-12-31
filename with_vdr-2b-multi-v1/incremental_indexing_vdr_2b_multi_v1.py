# Standard library imports
import os
import uuid
from datetime import datetime
from pathlib import Path

# Third-party imports
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access environment variables with defaults
indexing_model_name = os.getenv('vdr_2b_multi_v1')
collection_name = os.getenv('collection_name_vdr')

class VDRIndexer:
    """
    Visual Document Retrieval Indexer class for processing and indexing document images.
    
    This class handles the creation and maintenance of a vector database for document
    images using the VDR-2B-Multi-V1 model for embedding generation.
    """
    
    def __init__(self, collection_name=collection_name):
        """
        Initialize the VDR indexer with necessary components.
        
        Args:
            collection_name (str): Name of the Qdrant collection to use
        """
        # Set collection name
        self.collection_name = collection_name
        
        # Initialize VDR model with GPU if available
        self.model = HuggingFaceEmbedding(
            model_name=indexing_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Initialize Qdrant client for vector storage
        self.client = QdrantClient("localhost", port=6333)
        self.vector_size = 1536  # VDR model embedding size
        
        # Initialize processing statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Store the base path for relative path calculations
        self.base_path = None

    def ensure_collection_exists(self):
        """
        Create the vector collection if it doesn't exist, with optimal parameters.
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not exists:
                # Configure vector parameters
                vector_params = models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
                
                # Create new collection with optimized settings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=100
                    ),
                    on_disk_payload=True
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def scan_folders(self, root_folders):
        """
        Recursively scan folders for supported image files.
        
        Args:
            root_folders (list): List of root folder paths to scan
        
        Returns:
            list: List of all found image paths
        """
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Find common base path for relative path calculations
        self.base_path = str(Path(os.path.commonpath(root_folders)))
        logger.info(f"Using common base path: {self.base_path}")
        
        # Scan each root folder
        for root_folder in root_folders:
            root_path = Path(root_folder)
            logger.info(f"Scanning folder: {root_path}")
            
            try:
                # Recursively find all image files
                for file_path in root_path.rglob('*'):
                    if file_path.suffix.lower() in image_extensions:
                        image_paths.append(str(file_path))
            except Exception as e:
                logger.error(f"Error scanning folder {root_folder}: {e}")
        
        logger.info(f"Found {len(image_paths)} images in total")
        return image_paths

    def process_batch(self, batch_paths):
        """
        Process a batch of images, handling any errors.
        
        Args:
            batch_paths (list): List of image paths to process
            
        Returns:
            tuple: (list of loaded images, list of valid paths)
        """
        images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                # Load and convert image to RGB
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                self.stats['failed'] += 1
                continue
        
        return images, valid_paths

    def index_folders(self, folder_paths, batch_size=16):
        """
        Index images from multiple folders with batch processing.
        
        Args:
            folder_paths (list): List of folder paths to index
            batch_size (int): Number of images to process at once
        """
        # Reset processing statistics
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        
        # Ensure collection exists
        self.ensure_collection_exists()
        
        # Get all image paths
        image_paths = self.scan_folders(folder_paths)
        if not image_paths:
            logger.warning("No images found to process")
            return
        
        # Process images in batches
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Iterate through batches with progress bar
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Process current batch
            images, valid_paths = self.process_batch(batch_paths)
            if not images:
                continue
            
            try:
                # Generate embeddings for batch
                batch_embeddings = []
                for img_path in valid_paths:
                    embedding = self.model.get_image_embedding(img_path)
                    batch_embeddings.append(embedding)
                
                # Prepare points for database insertion
                points = []
                for j, embedding in enumerate(batch_embeddings):
                    path = valid_paths[j]
                    # Create relative path for storage
                    rel_path = str(Path(path).relative_to(self.base_path))
                    
                    # Create point with metadata
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                'filename': Path(path).name,
                                'relative_path': rel_path,
                                'full_path': path,
                                'folder': str(Path(path).parent),
                                'indexed_at': datetime.now().isoformat(),
                            }
                        )
                    )
                
                # Upload batch to database
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                
                self.stats['processed'] += len(points)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                self.stats['failed'] += len(batch_paths)
                continue

        # Print final processing statistics
        logger.info("Indexing completed!")
        logger.info(f"Processed: {self.stats['processed']} images")
        logger.info(f"Failed: {self.stats['failed']} images")
        logger.info(f"Skipped: {self.stats['skipped']} images")
        
        # Get final collection statistics
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"Total points in collection: {collection_info.points_count}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add shared folder to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
    try:
        from get_all_folder_details import get_subfolder_paths
    except ImportError:
        def get_subfolder_paths(main_folder):
            subfolder_paths = []
            for item in os.listdir(main_folder):
                full_path = os.path.join(main_folder, item)
                if os.path.isdir(full_path):
                    subfolder_paths.append(str(Path(full_path)))
            return subfolder_paths
    
    # Get folder paths from environment or discover them automatically
    converted_images_folder = os.getenv('converted_images_folder')
    
    if converted_images_folder:
        folders_to_index = get_subfolder_paths(converted_images_folder)
        logger.info(f"Auto-discovered {len(folders_to_index)} folders to index")
    else:
        # Fallback: specify folders manually
        folders_to_index = [
            # Add your folder paths here, or set converted_images_folder in .env
            # e.g., "D:/path/to/images/folder1",
        ]
    
    if not folders_to_index:
        logger.error(
            "No folders to index. Either:\n"
            "1. Set 'converted_images_folder' in .env to auto-discover subfolders, or\n"
            "2. Add folder paths to the 'folders_to_index' list in this script."
        )
        sys.exit(1)
    
    # Initialize and run indexer
    indexer = VDRIndexer(collection_name=collection_name)
    indexer.index_folders(folders_to_index)