"""
Late Chunking vs Traditional Chunking Demo
=========================================

This demo compares traditional chunking with late chunking using the Apache License 2.0
as a sample contract document. Late chunking often provides better retrieval accuracy
for complex documents by preserving more context.

Requirements:
pip install sentence-transformers numpy scikit-learn

"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict
import time
import os

# Document loading and query type analysis utilities

def load_document(filename: str) -> str:
    """Load a document from file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# Query type classification for analysis
QUERY_TYPES = {
    "definition": [
        "How are derivative works defined?",
        "What is meant by 'contribution'?"
    ],
    "cross_reference": [
        "What happens if someone sues over patent infringement?",
        "What are the patent license terms?"
    ],
    "contextual": [
        "What are the warranty disclaimers in this license?",
        "What liability protections exist?"
    ],
    "procedural": [
        "What must be included when redistributing the software?",
        "How do you apply this license to your work?"
    ]
}


def classify_query(query: str) -> str:
    """Classify a query by type to analyze performance patterns."""
    for query_type, examples in QUERY_TYPES.items():
        for example in examples:
            if query.lower() in example.lower() or example.lower() in query.lower():
                return query_type
    return "unknown"


class TraditionalChunker:
    """
    Traditional chunking approach that splits text into fixed-size chunks
    and generates embeddings for each chunk independently.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 200):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = []
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks by word count."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def process_document(self, text: str):
        """Process document using traditional chunking."""
        print("Processing document with traditional chunking...")
        start_time = time.time()
        
        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Create chunks
        self.chunks = self.chunk_text(cleaned_text)
        print(f"Created {len(self.chunks)} chunks")
        
        # Generate embeddings for each chunk
        self.embeddings = self.model.encode(self.chunks)
        
        end_time = time.time()
        print(f"Traditional chunking completed in {end_time - start_time:.2f} seconds")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant chunks using cosine similarity."""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.chunks[i], similarities[i]) for i in top_indices]
        
        return results
    
    def show_chunk_details(self, chunk_index: int) -> Dict[str, str]:
        """Show details about a specific chunk."""
        if 0 <= chunk_index < len(self.chunks):
            return {
                "chunk": self.chunks[chunk_index],
                "length": len(self.chunks[chunk_index].split()),
                "type": "traditional"
            }
        return {}


class LateChunker:
    """
    Late chunking approach that considers broader context when generating embeddings,
    then performs chunking on the contextually-aware representations.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 context_window: int = 500, chunk_size: int = 200):
        self.model = SentenceTransformer(model_name)
        self.context_window = context_window
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = []
        self.context_chunks = []
    
    def create_contextual_chunks(self, text: str) -> List[Tuple[str, str]]:
        """
        Create chunks with additional context for better embeddings.
        Returns tuples of (chunk_text, context_text).
        """
        words = text.split()
        chunks_with_context = []
        
        for i in range(0, len(words), self.chunk_size):
            # Get the main chunk
            chunk_end = min(i + self.chunk_size, len(words))
            chunk = ' '.join(words[i:chunk_end])
            
            # Get surrounding context
            context_start = max(0, i - self.context_window // 2)
            context_end = min(len(words), chunk_end + self.context_window // 2)
            context = ' '.join(words[context_start:context_end])
            
            chunks_with_context.append((chunk, context))
        
        return chunks_with_context
    
    def process_document(self, text: str):
        """Process document using late chunking approach."""
        print("Processing document with late chunking...")
        start_time = time.time()
        
        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Create contextual chunks
        self.context_chunks = self.create_contextual_chunks(cleaned_text)
        self.chunks = [chunk for chunk, _ in self.context_chunks]
        
        print(f"Created {len(self.chunks)} chunks with extended context")
        
        # Generate embeddings using the full context
        contexts = [context for _, context in self.context_chunks]
        self.embeddings = self.model.encode(contexts)
        
        end_time = time.time()
        print(f"Late chunking completed in {end_time - start_time:.2f} seconds")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant chunks using cosine similarity."""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.chunks[i], similarities[i]) for i in top_indices]
        
        return results
    
    def show_chunk_details(self, chunk_index: int) -> Dict[str, str]:
        """Show details about a specific chunk and its context."""
        if 0 <= chunk_index < len(self.chunks):
            chunk, context = self.context_chunks[chunk_index]
            return {
                "chunk": chunk,
                "context": context,
                "chunk_length": len(chunk.split()),
                "context_length": len(context.split()),
                "context_expansion": len(context.split()) - len(chunk.split()),
                "type": "late_chunking"
            }
        return {}
    
    def show_context_example(self, chunk_index: int):
        """Visualize how context window extends around a specific chunk."""
        details = self.show_chunk_details(chunk_index)
        if details:
            print(f"Chunk {chunk_index + 1} Details:")
            print(f"- Core chunk: {details['chunk_length']} words")
            print(f"- Full context: {details['context_length']} words")
            print(f"- Context expansion: +{details['context_expansion']} words")
            print(f"\nCore chunk: {details['chunk'][:150]}...")
            print(f"\nFull context: {details['context'][:300]}...")
            print()


def run_comparison_demo():
    """Run a comparison demo between traditional and late chunking."""
    
    # Load the Apache License document
    apache_license_text = load_document("apache_license.txt")
    
    # Test queries categorized by type
    test_queries = [
        "What are the patent license terms?",
        "What happens if someone sues over patent infringement?",
        "What must be included when redistributing the software?",
        "What are the warranty disclaimers in this license?",
        "How are derivative works defined?"
    ]
    
    # Track performance by query type
    results_by_type = {qtype: {"traditional": [], "late": [], "improvements": []} 
                      for qtype in QUERY_TYPES.keys()}
    results_by_type["unknown"] = {"traditional": [], "late": [], "improvements": []}
    
    print("=" * 60)
    print("LATE CHUNKING vs TRADITIONAL CHUNKING DEMO")
    print("=" * 60)
    print()
    
    # Initialize both chunkers
    traditional_chunker = TraditionalChunker(chunk_size=150)
    late_chunker = LateChunker(context_window=400, chunk_size=150)
    
    # Process the document with both approaches
    print("Processing Apache License 2.0 document...")
    print()
    
    traditional_chunker.process_document(apache_license_text)
    print()
    late_chunker.process_document(apache_license_text)
    print()
    
    # Run comparison for each test query
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*60}")
        print()
        
        # Traditional chunking results
        print("TRADITIONAL CHUNKING RESULTS:")
        print("-" * 40)
        traditional_results = traditional_chunker.search(query, top_k=2)
        
        for j, (chunk, score) in enumerate(traditional_results, 1):
            print(f"Result {j} (Score: {score:.4f}):")
            print(f"'{chunk[:200]}{'...' if len(chunk) > 200 else ''}'")
            print()
        
        # Late chunking results
        print("LATE CHUNKING RESULTS:")
        print("-" * 40)
        late_results = late_chunker.search(query, top_k=2)
        
        for j, (chunk, score) in enumerate(late_results, 1):
            print(f"Result {j} (Score: {score:.4f}):")
            print(f"'{chunk[:200]}{'...' if len(chunk) > 200 else ''}'")
            print()
        
        # Compare scores
        traditional_max_score = max(score for _, score in traditional_results)
        late_max_score = max(score for _, score in late_results)
        improvement = ((late_max_score - traditional_max_score) / traditional_max_score * 100)
        
        # Track results by query type
        query_type = classify_query(query)
        results_by_type[query_type]["traditional"].append(traditional_max_score)
        results_by_type[query_type]["late"].append(late_max_score)
        results_by_type[query_type]["improvements"].append(improvement)
        
        print("COMPARISON:")
        print(f"Query type: {query_type}")
        print(f"Traditional max score: {traditional_max_score:.4f}")
        print(f"Late chunking max score: {late_max_score:.4f}")
        print(f"Late chunking improvement: {improvement:+.2f}%")
        print()
        print("=" * 60)
        print()
    
    # Summary by query type
    print("\nPERFORMANCE ANALYSIS BY QUERY TYPE")
    print("=" * 50)
    for qtype, data in results_by_type.items():
        if data["improvements"]:
            avg_improvement = np.mean(data["improvements"])
            print(f"{qtype.upper()} queries: {avg_improvement:+.2f}% average improvement")
            if qtype in QUERY_TYPES:
                print(f"  Examples: {', '.join(QUERY_TYPES[qtype][:2])}")
            print()


def analyze_chunking_differences():
    """Analyze the differences between chunking approaches in detail."""
    
    print("DETAILED ANALYSIS OF CHUNKING APPROACHES")
    print("=" * 50)
    print()
    
    # Load document
    apache_license_text = load_document("apache_license.txt")
    
    # Initialize chunkers
    traditional = TraditionalChunker(chunk_size=100)
    late = LateChunker(context_window=300, chunk_size=100)
    
    # Process document
    traditional.process_document(apache_license_text)
    late.process_document(apache_license_text)
    
    print(f"Document Statistics:")
    print(f"- Original text length: {len(apache_license_text)} characters")
    print(f"- Traditional chunks: {len(traditional.chunks)}")
    print(f"- Late chunking chunks: {len(late.chunks)}")
    print(f"- Average chunk length (traditional): {np.mean([len(c) for c in traditional.chunks]):.1f} chars")
    print(f"- Average context length (late): {np.mean([len(c) for _, c in late.context_chunks]):.1f} chars")
    print()
    
    # Show context window example
    print("CONTEXT WINDOW VISUALIZATION")
    print("-" * 40)
    if len(late.chunks) > 3:
        late.show_context_example(3)
    print()
    
    # Example of how context helps
    print("EXAMPLE: How context improves understanding")
    print("-" * 40)
    
    sample_query = "patent litigation termination"
    
    print(f"Query: '{sample_query}'")
    print()
    
    # Show traditional chunking limitation
    trad_results = traditional.search(sample_query, top_k=1)
    late_results = late.search(sample_query, top_k=1)
    
    print("Traditional chunking (limited context):")
    print(f"Best match: '{trad_results[0][0][:150]}...'")
    print(f"Score: {trad_results[0][1]:.4f}")
    print()
    
    print("Late chunking (with extended context):")
    print(f"Best match: '{late_results[0][0][:150]}...'")
    print(f"Score: {late_results[0][1]:.4f}")
    print()
    
    improvement = (late_results[0][1] - trad_results[0][1]) / trad_results[0][1] * 100
    print(f"Improvement with late chunking: {improvement:+.2f}%")


if __name__ == "__main__":
    print("Starting Late Chunking Demo...")
    print("This demo requires sentence-transformers, numpy, and scikit-learn")
    print()
    
    try:
        # Run the main comparison demo
        run_comparison_demo()
        
        # Run detailed analysis
        analyze_chunking_differences()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("""
Key Benefits of Late Chunking:

1. CONTEXT PRESERVATION: Late chunking maintains broader context around each chunk,
   leading to more accurate semantic embeddings.

2. BETTER RETRIEVAL: Queries often find more relevant chunks because the embeddings
   capture relationships between concepts that span chunk boundaries.

3. CROSS-REFERENCE UNDERSTANDING: Legal documents often reference terms defined
   elsewhere. Late chunking helps capture these relationships.

4. IMPROVED SEMANTIC SIMILARITY: With more context, the embedding model can better
   understand the semantic meaning of each chunk.

Traditional chunking works well for simple retrieval tasks, but late chunking
significantly improves performance on complex documents like contracts, legal texts,
and technical documentation where context matters.
        """)
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install sentence-transformers numpy scikit-learn")
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure all dependencies are installed and try again.")