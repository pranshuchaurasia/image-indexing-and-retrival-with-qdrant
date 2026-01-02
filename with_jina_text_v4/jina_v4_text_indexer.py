# Standard library imports
import os
import uuid
from datetime import datetime

# Third-party imports
import torch
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoModel
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

indexing_model_name = os.getenv('jinaai_jina_embeddings_v4')
# Using the specific collection name for text (dense vectors)
collection_name = os.getenv('collection_name_jinaai_text', 'jina_v4_text_dense')

class JinaV4TextIndexer:
    """
    Jina Embeddings v4 Text Indexer using Single-vector (Dense) strategy.
    
    Why Single-vector?
    For standard text-to-text retrieval, single dense vectors (2048d) often provide 
    a great balance of performance and efficiency. They are faster to index and query 
    compared to multivectors, especially for large datasets. Jina V4 excels at this 
    mode as well, providing high-quality semantic representations of text blocks.
    """
    
    def __init__(self, collection_name=collection_name):
        self.collection_name = collection_name
        
        # Initialize Model
        logger.info(f"Loading model from {indexing_model_name}...")
        self.model = AutoModel.from_pretrained(
            indexing_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize Qdrant
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(host, port=port)
        
        # Jina V4 Single-vector Dimension is 2048
        self.vector_size = 2048
        
        self.stats = {'processed': 0, 'failed': 0}

    def ensure_collection_exists(self):
        """
        Create standard Qdrant collection for dense vectors.
        """
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not exists:
                vector_params = models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                    # No multivector_config here -> Standard Dense Vector
                )
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                    on_disk_payload=True
                )
                logger.info(f"Created new Dense collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def index_texts(self, texts, metadata_list=None, batch_size=8):
        """
        Index a list of text strings.
        
        Args:
            texts (list): List of strings to index
            metadata_list (list): List of dicts corresponding to texts
        """
        self.ensure_collection_exists()
        
        if not metadata_list:
            metadata_list = [{}] * len(texts)
            
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing texts"):
            batch_texts = texts[i:i+batch_size]
            batch_meta = metadata_list[i:i+batch_size]
            
            try:
                # Generate Single-vector Embeddings
                # return_multivector=False (default) gives us one 2048d vector per text
                # task="retrieval" and prompt_name="passage" is standard for indexing documents
                embeddings = self.model.encode_text(
                    texts=batch_texts,
                    task="retrieval",
                    prompt_name="passage"
                )
                
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                
                points = []
                for j, vector in enumerate(embeddings):
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                'content': batch_texts[j],
                                'indexed_at': datetime.now().isoformat(),
                                'strategy': 'single_vector_jina_v4',
                                **batch_meta[j]
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
                logger.error(f"Error indexing batch: {e}")
                self.stats['failed'] += len(batch_texts)

        logger.info(f"Indexing completed. Processed: {self.stats['processed']}")

if __name__ == "__main__":
    import json
    
    # Path to sample data
    json_path = os.path.join(os.path.dirname(__file__), 'sample_text_data.json')
    
    if os.path.exists(json_path):
        logger.info(f"Loading data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        texts_to_index = [item['content'] for item in data]
        metadata_list = [{'title': item['title'], 'source_id': item['id']} for item in data]
        
        logger.info(f"Found {len(texts_to_index)} documents to index.")
        
        indexer = JinaV4TextIndexer()
        indexer.index_texts(texts_to_index, metadata_list=metadata_list)
    else:
        logger.warning(f" Sample data not found at {json_path}. creating dummy data...")
        texts_to_index = [
            "Jina Embeddings v4 is a powerful multimodal model.",
            "Qdrant is a high-performance vector database.",
            "Multivectors allow for late interaction and fine-grained retrieval.",
            "Single vectors are efficient for large-scale semantic search."
        ]
        indexer = JinaV4TextIndexer()
        indexer.index_texts(texts_to_index)
