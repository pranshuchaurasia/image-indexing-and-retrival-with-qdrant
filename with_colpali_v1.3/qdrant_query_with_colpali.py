# Standard library imports
import os
import sys
import time
from pathlib import Path

# Third-party imports
import torch
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# ColPali imports
from colpali_engine.models import ColPali, ColPaliProcessor

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access environment variables
colpali_processor_path = os.getenv('colpali_processor')
colpali_model_path = os.getenv('colpali_model')
collection_name = os.getenv('collection_name_colpali', 'colpali_annual_report')


class ColPaliSearcher:
    """
    ColPali Document Searcher for querying indexed documents.
    
    Uses multi-vector late interaction scoring (MaxSim) for retrieval.
    """
    
    def __init__(self, model_path=None, processor_path=None):
        """
        Initialize the ColPali searcher.
        
        Args:
            model_path (str): Path to ColPali model (overrides env var)
            processor_path (str): Path to ColPali processor (overrides env var)
        """
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
        
        # Initialize database client
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

    def get_query_embedding(self, query_text: str):
        """Generate multi-vector embedding for a query text."""
        batch_queries = self.processor.process_queries([query_text])
        batch_queries = {k: v.to(self.device) for k, v in batch_queries.items()}
        
        with torch.no_grad():
            query_embedding = self.model(**batch_queries)
        
        return query_embedding[0].cpu().float().numpy().tolist()

    def search(self, query_text: str, top_k: int = 5):
        """
        Search for documents matching the query text.
        
        Args:
            query_text (str): The search query
            top_k (int): Number of results to return
            
        Returns:
            Search results with scores and payloads
        """
        try:
            query_vector = self.get_query_embedding(query_text)
            
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                timeout=6000
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return None


def format_results(search_result):
    """Format search results for display."""
    if not search_result:
        return []
        
    results = []
    for point in search_result.points:
        result = {
            'id': point.id,
            'score': point.score,
            'filename': point.payload.get('filename', 'Unknown'),
            'path': point.payload.get('relative_path', 'Unknown'),
            'full_path': point.payload.get('full_path', 'Unknown')
        }
        results.append(result)
    return results


if __name__ == "__main__":
    # Initialize searcher
    searcher = ColPaliSearcher()
    
    # Example queries
    example_queries = [
        "What was the total revenue in 2023?",
        "Show me the balance sheet",
        "What are the key financial highlights?"
    ]
    
    print("Running example searches...")
    for query in example_queries:
        start_time = time.perf_counter()
        print(f"\nSearch query: '{query}'")
        
        results = searcher.search(query)
        formatted_results = format_results(results)
        
        print(f"Found {len(formatted_results)} results:")
        for i, result in enumerate(formatted_results, 1):
            print(f"{i}. Score: {result['score']:.3f} | File: {result['filename']}")
            print(f"   Path: {result['full_path']}")
            
        execution_time = time.perf_counter() - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
