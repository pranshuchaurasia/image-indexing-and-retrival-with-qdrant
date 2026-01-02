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

load_dotenv()

model_name = os.getenv('jinaai_jina_embeddings_v4')
collection_name = os.getenv('collection_name_jinaai_text', 'jina_v4_text_dense')

class JinaV4TextSearcher:
    """
    Jina Embeddings v4 Text Searcher using Single-vector (Dense) strategy.
    """
    
    def __init__(self):
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
        Search for texts matching the query using Dense Vectors.
        """
        try:
            # Generate Single-vector Embedding for Query
            # task="retrieval" and prompt_name="query"
            query_vector = self.model.encode_text(
                texts=[query_text],
                task="retrieval",
                prompt_name="query"
                # return_multivector=False by default
            )[0]
            
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            
            # Standard Cosine Similarity Search
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
    if not search_result:
        return []
    results = []
    for point in search_result.points:
        result = {
            'id': point.id,
            'score': point.score,
            'content': point.payload.get('content', 'Unknown'),
        }
        results.append(result)
    return results

if __name__ == "__main__":
    searcher = JinaV4TextSearcher()
    
    example_queries = [
        "What is Jina V4?",
        "How does Qdrant work?"
    ]
    
    print("Running example searches (Single-vector/Dense)...")
    for query in example_queries:
        start_time = time.perf_counter()
        print(f"\nSearch query: '{query}'")
        
        results = searcher.search(query)
        formatted_results = format_results(results)
        
        print(f"Found {len(formatted_results)} results:")
        for i, result in enumerate(formatted_results, 1):
            print(f"{i}. Score: {result['score']:.3f} | Content: {result['content']}")
            
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
