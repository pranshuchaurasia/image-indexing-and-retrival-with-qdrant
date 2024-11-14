import os
import torch
import numpy as np
from datetime import datetime
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from colpali_engine.utils.colpali_processing_utils import process_images
import logging
import os
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.utils.colpali_processing_utils import process_images
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColPaliIndexer:
    def __init__(self, collection_name="my_document_collection"):
        """
        Initialize the ColPali indexer
        
        Args:
            collection_name (str): Name of the Qdrant collection to use
        """
        # Set collection name
        self.collection_name = collection_name
        
        # Initialize model and processor
        self.model_name = r"model_path"
        self.model = ColPali.from_pretrained(
            r"paligemma_path",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        ).eval()
        
        self.model.load_adapter(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Initialize Qdrant client
        self.client = QdrantClient("localhost", port=6333)
        self.vector_size = 128
        
        # Track processing statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Store the base path for all folders
        self.base_path = None

    def ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not exists:
                vector_params = models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
                
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
        Recursively scan folders for images
        
        Args:
            root_folders (list): List of root folder paths to scan
        
        Returns:
            list: List of all found image paths
        """
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Find the common base path
        self.base_path = str(Path(os.path.commonpath(root_folders)))
        logger.info(f"Using common base path: {self.base_path}")
        
        for root_folder in root_folders:
            root_path = Path(root_folder)
            logger.info(f"Scanning folder: {root_path}")
            
            try:
                # Recursively scan for images
                for file_path in root_path.rglob('*'):
                    if file_path.suffix.lower() in image_extensions:
                        image_paths.append(str(file_path))
            except Exception as e:
                logger.error(f"Error scanning folder {root_folder}: {e}")
        
        logger.info(f"Found {len(image_paths)} images in total")
        return image_paths

    def process_batch(self, batch_paths):
        """Process a batch of images"""
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

    def index_folders(self, folder_paths, batch_size=1):
        """
        Index images from multiple folders
        
        Args:
            folder_paths (list): List of folder paths to index
            batch_size (int): Batch size for processing
        """
        # Reset statistics
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        
        # Ensure collection exists
        self.ensure_collection_exists()
        
        # Scan all folders for images
        image_paths = self.scan_folders(folder_paths)
        if not image_paths:
            logger.warning("No images found to process")
            return
        
        # Process images in batches
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != self.model.device:
            self.model.to(device)

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Process batch
            images, valid_paths = self.process_batch(batch_paths)
            if not images:
                continue
            
            # Create embeddings
            try:
                batch_doc = process_images(self.processor, images)
                batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
                
                with torch.no_grad():
                    image_embeddings = self.model(**batch_doc)
                
                # Prepare points for Qdrant
                points = []
                for j, embedding in enumerate(image_embeddings):
                    path = valid_paths[j]
                    # Use the common base path to create relative paths
                    rel_path = str(Path(path).relative_to(self.base_path))
                    
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding.cpu().float().numpy().tolist(),
                            payload={
                                'filename': Path(path).name,
                                'relative_path': rel_path,
                                'full_path': path,
                                'folder': str(Path(path).parent),
                                'indexed_at': datetime.now().isoformat(),
                            }
                        )
                    )
                
                # Upload to Qdrant
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

        # Print final statistics
        logger.info("Indexing completed!")
        
        logger.info(f"Processed: {self.stats['processed']} images")
        logger.info(f"Failed: {self.stats['failed']} images")
        logger.info(f"Skipped: {self.stats['skipped']} images")
        
        # Get collection info
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"Total points in collection: {collection_info.points_count}")
# Example usage
if __name__ == "__main__":
    # Specify your folders and collection name
    folders_to_index =[r"Folders List to index"]
    
    # Initialize indexer with custom collection name
    indexer = ColPaliIndexer(collection_name="collection_name")
    
    # Index all folders
    indexer.index_folders(folders_to_index)