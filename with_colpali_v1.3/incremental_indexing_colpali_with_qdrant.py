# Standard library imports
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Third-party imports
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from qdrant_client import QdrantClient
from qdrant_client.http import models

from colpali_engine.models import ColPali, ColPaliProcessor
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access environment variables with defaults
colpali_processor_path = os.getenv('colpali_processor')
colpali_model_path = os.getenv('colpali_model')
collection_name = os.getenv('collection_name_colpali', 'colpali_annual_report')
converted_images_folder = os.getenv('converted_images_folder')

# Add shared folder to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
try:
    from get_all_folder_details import get_subfolder_paths
except ImportError:
    # Fallback if shared module not found
    def get_subfolder_paths(main_folder):
        subfolder_paths = []
        for item in os.listdir(main_folder):
            full_path = os.path.join(main_folder, item)
            if os.path.isdir(full_path):
                subfolder_paths.append(str(Path(full_path)))
        return subfolder_paths


class ColPaliIndexer:
    """
    ColPali Document Indexer class for processing and indexing document images.
    
    Uses ColPali model for multi-vector embedding generation (1030 x 128D per image)
    which enables fine-grained visual document retrieval with late interaction scoring.
    """
    
    def __init__(self, collection_name=collection_name, 
                 model_path=None, processor_path=None):
        """
        Initialize the ColPali indexer.
        
        Args:
            collection_name (str): Name of the Qdrant collection
            model_path (str): Path to ColPali model (overrides env var)
            processor_path (str): Path to ColPali processor (overrides env var)
        """
        self.collection_name = collection_name
        
        # Use provided paths or fall back to environment variables
        self.model_path = model_path or colpali_model_path
        self.processor_path = processor_path or colpali_processor_path
        
        if not self.model_path or not self.processor_path:
            raise ValueError(
                "Model paths not configured. Set colpali_model and colpali_processor "
                "in .env file or pass them as arguments."
            )
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Set dtype based on device
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Initialize ColPali model
        logger.info(f"Loading ColPali model from: {self.model_path}")
        self.model = ColPali.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            local_files_only=True,
        ).eval()
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        # Initialize ColPali processor
        logger.info(f"Loading ColPali processor from: {self.processor_path}")
        self.processor = ColPaliProcessor.from_pretrained(
            self.processor_path,
            local_files_only=True,
        )
        
        # Initialize Qdrant client
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.vector_size = 128  # ColPali embedding dimension
        
        # Processing statistics
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        self.base_path = None

    def ensure_collection_exists(self):
        """Create the vector collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),
                    on_disk_payload=True
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def scan_folders(self, root_folders):
        """Recursively scan folders for supported image files."""
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
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

    def load_image(self, path):
        """Load a single image safely."""
        try:
            img = Image.open(path).convert('RGB')
            return img, path
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            self.stats['failed'] += 1
            return None, path

    def collate_fn(self, batch):
        """Custom collate function for DataLoader."""
        images = []
        valid_paths = []
        
        for img, path in batch:
            if img is not None:
                images.append(img)
                valid_paths.append(path)
        
        if not images:
            return None, []
        
        processed = self.processor.process_images(images)
        return processed, valid_paths

    def upsert_to_qdrant(self, points):
        """Upload points to Qdrant collection."""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            return True
        except Exception as e:
            logger.error(f"Error during upsert: {e}")
            return False

    def index_folders(self, folder_paths, batch_size=16):
        """
        Index images from multiple folders.
        
        Args:
            folder_paths (list): List of folder paths to index
            batch_size (int): Number of images per batch (default 16)
        """
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        self.ensure_collection_exists()
        
        image_paths = self.scan_folders(folder_paths)
        if not image_paths:
            logger.warning("No images found to process")
            return

        class ImageDataset:
            def __init__(self, paths, loader):
                self.paths = paths
                self.loader = loader
            
            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                return self.loader(self.paths[idx])
        
        dataset = ImageDataset(image_paths, self.load_image)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )
        
        for batch_doc, valid_paths in tqdm(dataloader, desc="Processing batches"):
            if batch_doc is None or not valid_paths:
                continue
            
            try:
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                
                with torch.no_grad():
                    image_embeddings = self.model(**batch_doc)
                
                points = []
                for j, embedding in enumerate(image_embeddings):
                    path = valid_paths[j]
                    rel_path = str(Path(path).relative_to(self.base_path))
                    multivector = embedding.cpu().float().numpy().tolist()
                    
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=multivector,
                            payload={
                                'filename': Path(path).name,
                                'relative_path': rel_path,
                                'full_path': path,
                                'folder': str(Path(path).parent),
                                'indexed_at': datetime.now().isoformat(),
                                'source': 'document'
                            }
                        )
                    )
                
                if self.upsert_to_qdrant(points):
                    self.stats['processed'] += len(points)
                else:
                    self.stats['failed'] += len(valid_paths)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                self.stats['failed'] += len(valid_paths)
                continue

        logger.info("Indexing completed!")
        logger.info(f"Processed: {self.stats['processed']} images")
        logger.info(f"Failed: {self.stats['failed']} images")
        
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"Total points in collection: {collection_info.points_count}")


if __name__ == "__main__":
    # Get folder paths from environment or discover them automatically
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
    indexer = ColPaliIndexer(collection_name=collection_name)
    indexer.index_folders(folders_to_index, batch_size=16)
