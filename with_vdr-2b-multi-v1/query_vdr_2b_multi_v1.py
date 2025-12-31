# Standard library imports
import os
import time

# Third-party imports
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access environment variables with defaults
model_name = os.getenv('vdr_2b_multi_v1', 'Xenova/vdr-2b-multi-v1')
collection_name = os.getenv('collection_name_vdr', 'vdr_documents_v1')

class VDRSearcher:
    """
    Visual Document Retrieval Searcher class for querying indexed documents.
    
    This class provides functionality to search through indexed document images
    using natural language queries across multiple languages.
    """
    
    def __init__(self):
        """
        Initialize the VDR searcher with model and database connection.
        """
        # Initialize model with GPU if available
        # logic: HuggingFaceEmbedding fails with this custom model due to tokenization issues
        # switching to direct SentenceTransformer usage which is verified to work
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Initialize database client
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name

    def search(self, query_text: str, top_k: int = 5):
        """
        Search for documents matching the query text.
        
        Args:
            query_text (str): The search query in any supported language
            top_k (int): Number of results to return
            
        Returns:
            list: List of search results with scores and payloads
            
        Example:
            >>> searcher = VDRSearcher()
            >>> results = searcher.search("What was the total revenue in 2023?", top_k=3)
        """
        try:
            # Generate embedding for query text
            # This custom model returns a dict containing 'sentence_embedding'
            embedding_output = self.model.encode(query_text)
            
            if isinstance(embedding_output, dict) and 'sentence_embedding' in embedding_output:
                query_vector = embedding_output['sentence_embedding']
            else:
                query_vector = embedding_output

            # Ensure we have a list of floats for Qdrant
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            
            # Search in vector database
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                timeout=6000  # 6 second timeout
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return None

def format_results(search_result):
    """
    Format search results for display.
    
    Args:
        search_result: Raw search results from Qdrant
        
    Returns:
        list: Formatted results with relevant metadata
        
    Example output:
        [
            {
                'id': 'uuid',
                'score': 0.95,
                'filename': 'page_1.png',
                'path': 'relative/path/to/file',
                'full_path': '/absolute/path/to/file'
            },
            ...
        ]
    """
    if not search_result:
        return []
        
    results = []
    for point in search_result.points:
        # Extract and format each result
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
    searcher = VDRSearcher()
    
    # Example queries demonstrating multi-lingual support
    example_queries = [
        "Query 1 ?",  # English
        "Query 2 ?",  # English
        "Query 3 ?"  # English
        # Add queries in other supported languages as needed
    ]
    
    # Run example searches
    print("Running example searches...")
    for query in example_queries:
        # Track execution time
        start_time = time.perf_counter()
        print(f"\nSearch query: '{query}'")
        
        # Perform search and format results
        results = searcher.search(query)
        formatted_results = format_results(results)
        
        # Display results
        print(f"Found {len(formatted_results)} results:")
        for i, result in enumerate(formatted_results, 1):
            print(f"{i}. Score: {result['score']:.3f} | File: {result['filename']}")
            print(f"   Path: {result['full_path']}")
            
        # Calculate and display execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")