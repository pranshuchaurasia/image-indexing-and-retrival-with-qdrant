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
from transformers import AutoModel
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access environment variables with defaults
indexing_model_name = os.getenv('jinaai_jina_embeddings_v4')
# Using the specific collection name for image multivectors
collection_name = os.getenv('collection_name_jinaai_image_multivector', 'jina_v4_image_colbert')

class JinaV4ImageIndexer:
    """
    Jina Embeddings v4 Image Indexer class using Multivector (ColBERT-like) strategy.
    
    Why Multivectors?
    Jina Embeddings V4 supports 'late interaction' or multivector representation.
    Instead of compressing an entire image into a single vector, it represents the image
    as a set of vectors (128-dimensional). This allows for much finer-grained matching,
    capturing small details in the image that might be lost in a single dense vector.
    This is particularly useful for complex documents, charts, or detailed scenes.
    """
    
    def __init__(self, collection_name=collection_name):
        """
        Initialize the Jina V4 Image indexer.
        """
        # Set collection name
        self.collection_name = collection_name
        
        # Initialize Jina V4 model
        # trust_remote_code=True is essential as Jina V4 uses custom modeling code.
        # torch_dtype=torch.float16 reduces memory usage on GPU.
        logger.info(f"Loading model from {indexing_model_name}...")
        self.model = AutoModel.from_pretrained(
            indexing_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize Qdrant client
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(host, port=port)
        
        # Jina V4 Multivector Dimension is 128
        # Unlike the single vector (2048), multivectors are smaller but numerous.
        self.vector_size = 128 
        
        # Processing stats
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        self.base_path = None

    def ensure_collection_exists(self):
        """
        Create the Qdrant collection with Multivector configuration.
        
        Why MultivectorConfig?
        Standard Qdrant collections expect one vector per point. 
        For multivectors (list of vectors), we must enable `multivector_config` 
        and use the `MAX_SIM` comparator. This tells Qdrant to compute the 
        MaxSim score (sum of maximum similarities) between the query vectors 
        and the stored document vectors, which is the standard for ColBERT-style retrieval.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not exists:
                # Define vector parameters with Multivector Config
                vector_params = models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
                
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                    # Optimize indexing threshold for performance
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=100
                    ),
                    on_disk_payload=True
                )
                logger.info(f"Created new Multivector collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def scan_folders(self, root_folders):
        """
        Recursively scan folders for images.
        """
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Determine common base path for relative path storage
        self.base_path = str(Path(os.path.commonpath(root_folders)))
        logger.info(f"Using common base path: {self.base_path}")
        
        for root_folder in root_folders:
            root_path = Path(root_folder)
            logger.info(f"Scanning folder: {root_path}")
            
            try:
                for file_path in root_path.rglob('*'):
                    if file_path.suffix.lower() in image_extensions:
                        image_paths.append(str(file_path))
            except Exception as e:
                logger.error(f"Error scanning folder {root_folder}: {e}")
        
        logger.info(f"Found {len(image_paths)} images in total")
        return image_paths

    def process_batch(self, batch_paths):
        """Load images for the batch."""
        images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                self.stats['failed'] += 1
                continue
        
        return images, valid_paths

    def index_folders(self, folder_paths, batch_size=4):
        """
        Index images using Multivector strategy.
        """
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        self.ensure_collection_exists()
        
        image_paths = self.scan_folders(folder_paths)
        if not image_paths:
            logger.warning("No images found to process")
            return
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            images, valid_paths = self.process_batch(batch_paths)
            if not images:
                continue
            
            try:
                # Generate Multivector Embeddings
                # We specify `return_multivector=True` to get the list of 128d vectors (ColBERT style).
                # `task="retrieval"` is the standard task for indexing documents.
                batch_embeddings = self.model.encode_image(
                    images=images,
                    task="retrieval",
                    return_multivector=True
                )
                
                # batch_embeddings is a list of tensors/arrays, where easy item is a matrix (N_tokens, 128)
                
                points = []
                for j, embedding in enumerate(batch_embeddings):
                    path = valid_paths[j]
                    rel_path = str(Path(path).relative_to(self.base_path))
                    
                    # Convert embedding to list of lists (required for Qdrant multivector)
                    if hasattr(embedding, 'tolist'):
                        vector_data = embedding.tolist()
                    else:
                        vector_data = embedding
                        
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector_data, # Use the list of vectors directly
                            payload={
                                'filename': Path(path).name,
                                'relative_path': rel_path,
                                'full_path': path,
                                'folder': str(Path(path).parent),
                                'indexed_at': datetime.now().isoformat(),
                                'strategy': 'multivector_jina_v4'
                            }
                        )
                    )
                
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

        # Final stats
        logger.info("Indexing completed!")
        logger.info(f"Processed: {self.stats['processed']} images")
        logger.info(f"Failed: {self.stats['failed']} images")
        
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"Total points in collection: {collection_info.points_count}")


if __name__ == "__main__":
    converted_images_root = os.getenv('converted_images_folder')
    if not converted_images_root or not os.path.exists(converted_images_root):
        logger.error(f"Converted images folder not found: {converted_images_root}")
        exit(1)

    # Scan root folder for subdirectories to mimic folder-based indexing
    folders_to_index = [f.path for f in os.scandir(converted_images_root) if f.is_dir()]
    if not folders_to_index:
        folders_to_index = [converted_images_root]
        
    logger.info(f"Found {len(folders_to_index)} folders to index")
    
    indexer = JinaV4ImageIndexer(collection_name=collection_name)
    indexer.index_folders(folders_to_index)
