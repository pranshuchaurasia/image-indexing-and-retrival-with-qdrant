# Standard library imports
import os
import time

# Third-party imports
import torch
from qdrant_client import QdrantClient
from transformers import AutoModel
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access environment variables
model_name = os.getenv('jinaai_jina_embeddings_v4')
# Must match the collection name used in indexing
collection_name = os.getenv('collection_name_jinaai_image_multivector', 'jina_v4_image_colbert')

class JinaV4ImageSearcher:
    """
    Jina Embeddings v4 Searcher class using Multivector strategy.
    
    This searcher is designed to work with the Multivector index created by jina_v4_image_indexer.py.
    """
    
    def __init__(self):
        """
        Initialize the Jina V4 searcher.
        """
        # Initialize Jina V4 model
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize Qdrant client
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(host, port=port)
        self.collection_name = collection_name

    def search(self, query_text: str, top_k: int = 5):
        """
        Search for images matching the query text using Multivector (MaxSim).
        
        Args:
            query_text (str): search query
            top_k (int): results count
        """
        try:
            # Generate Multivector Embedding for Query
            # Critical: We must use `return_multivector=True` for the query as well.
            # MaxSim requires a set of vectors for the query to compare against the set of vectors for the document.
            # prompt_name="query" is standard for query encoding in Jina V4.
            query_embeddings = self.model.encode_text(
                texts=[query_text],
                task="retrieval",
                prompt_name="query",
                return_multivector=True
            )[0] # Take first result (list of vectors)
            
            # Convert to list of lists for Qdrant
            if hasattr(query_embeddings, 'tolist'):
                query_vector = query_embeddings.tolist()
            else:
                query_vector = query_embeddings
            
            # Search in vector database
            # Qdrant will automatically use the MaxSim comparator defined in the collection config
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
    """Format search results."""
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
    searcher = JinaV4ImageSearcher()
    
    # Example queries
    example_queries = [
        "What was the revenue of Alphabet in 2022?",
        "Explain the risk factors mentioned in Amazon's report.",
        "Compare the net income of Apple and Microsoft."
    ]
    
    print("Running example searches (Multivector/MaxSim)...")
    for query in example_queries:
        start_time = time.perf_counter()
        print(f"\nSearch query: '{query}'")
        
        results = searcher.search(query)
        formatted_results = format_results(results)
        
        print(f"Found {len(formatted_results)} results:")
        for i, result in enumerate(formatted_results, 1):
            print(f"{i}. Score: {result['score']:.3f} | File: {result['filename']}")
            
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
