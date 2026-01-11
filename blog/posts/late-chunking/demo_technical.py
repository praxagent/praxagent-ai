"""
Late Chunking vs Traditional Chunking Demo - Technical Documentation
===================================================================

This demo uses RFC 2616 (HTTP/1.1 specification) to demonstrate where late chunking
excels with technical documentation. RFCs are rich with cross-references, protocol
definitions, and procedural dependencies that benefit greatly from preserved context.

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


def load_document(filename: str) -> str:
    """Load a document from file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def clean_rfc_text(text: str) -> str:
    """Clean RFC text by removing headers/footers and normalizing."""
    lines = text.split('\n')
    
    # Remove RFC header boilerplate (usually first 50-100 lines)
    # Look for the actual start of content
    start_idx = 0
    for i, line in enumerate(lines[:200]):  # Check first 200 lines
        if any(keyword in line.lower() for keyword in ['table of contents', 'introduction', '1. introduction', 'abstract']):
            start_idx = i
            break
    
    # Remove page headers/footers and normalize
    cleaned_lines = []
    for line in lines[start_idx:]:
        # Skip RFC page headers/footers
        if re.match(r'^RFC \d+|^\s*\d+\.\d+|^\f|^Fielding.*Standards Track.*Page \d+', line):
            continue
        if re.match(r'^\s*HTTP/1\.1.*\d{4}$', line):  # Skip date headers
            continue
        if len(line.strip()) == 0:
            cleaned_lines.append('')  # Preserve paragraph breaks
        else:
            cleaned_lines.append(line.strip())
    
    content = '\n'.join(cleaned_lines)
    
    # Clean up excessive whitespace while preserving structure
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 line breaks
    content = re.sub(r'[ \t]+', ' ', content)            # Normalize spaces
    content = content.strip()
    
    return content


# Technical documentation query categories that benefit from context
TECHNICAL_QUERY_TYPES = {
    "protocol_methods": [
        "How does the GET method work?",
        "What happens with POST requests?",
        "How are HEAD requests processed?"
    ],
    "cross_references": [
        "What does this header do?",
        "How is this implemented?",
        "What are the requirements for this?"
    ],
    "procedural_flows": [
        "How does content negotiation work?",
        "What is the authentication process?",
        "How are errors handled?"
    ],
    "technical_definitions": [
        "What is a persistent connection?",
        "How are cookies handled?",
        "What is chunked encoding?"
    ]
}


def classify_technical_query(query: str) -> str:
    """Classify a technical query by type."""
    query_lower = query.lower()
    
    # Look for key patterns in queries
    if any(method in query_lower for method in ['get', 'post', 'head', 'put', 'delete']):
        return "protocol_methods"
    elif any(word in query_lower for word in ['this', 'that', 'these', 'such', 'the above']):
        return "cross_references"
    elif any(word in query_lower for word in ['how does', 'process', 'work', 'handled', 'flow']):
        return "procedural_flows"
    elif any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
        return "technical_definitions"
    
    return "unknown"


class TraditionalChunker:
    """Traditional chunking approach for technical documentation."""
    
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
        print("Processing with traditional chunking...")
        start_time = time.time()
        
        # Clean and chunk text
        cleaned_text = clean_rfc_text(text)
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


class LateChunker:
    """Late chunking approach optimized for technical cross-references."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 context_window: int = 800, chunk_size: int = 200):
        self.model = SentenceTransformer(model_name)
        self.context_window = context_window
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = []
        self.context_chunks = []
    
    def create_contextual_chunks(self, text: str) -> List[Tuple[str, str]]:
        """Create chunks with extended technical context for better cross-reference understanding."""
        words = text.split()
        chunks_with_context = []
        
        for i in range(0, len(words), self.chunk_size):
            # Get the main chunk
            chunk_end = min(i + self.chunk_size, len(words))
            chunk = ' '.join(words[i:chunk_end])
            
            # Get extended context for technical understanding
            # Technical docs benefit from larger context windows due to cross-references
            context_start = max(0, i - self.context_window // 2)
            context_end = min(len(words), chunk_end + self.context_window // 2)
            context = ' '.join(words[context_start:context_end])
            
            chunks_with_context.append((chunk, context))
        
        return chunks_with_context
    
    def process_document(self, text: str):
        """Process document using late chunking approach."""
        print("Processing with late chunking...")
        start_time = time.time()
        
        # Clean and create contextual chunks
        cleaned_text = clean_rfc_text(text)
        self.context_chunks = self.create_contextual_chunks(cleaned_text)
        self.chunks = [chunk for chunk, _ in self.context_chunks]
        
        print(f"Created {len(self.chunks)} chunks with extended technical context")
        
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
    
    def show_context_example(self, chunk_index: int):
        """Show how extended context helps with technical cross-references."""
        if 0 <= chunk_index < len(self.chunks):
            chunk, context = self.context_chunks[chunk_index]
            print(f"Technical Context Example - Chunk {chunk_index + 1}:")
            print(f"- Core chunk: {len(chunk.split())} words")
            print(f"- Full context: {len(context.split())} words")
            print(f"- Additional context: +{len(context.split()) - len(chunk.split())} words")
            print(f"\nCore chunk: {chunk[:300]}...")
            print(f"\nFull context: {context[:500]}...")
            print()


def run_technical_demo():
    """Run comparison demo using RFC 2616 (HTTP/1.1 specification)."""
    
    # Load RFC 2616
    print("Loading RFC 2616 (HTTP/1.1 Specification)...")
    rfc_text = load_document("rfc2616.txt")
    
    # Technical queries that should strongly favor late chunking
    test_queries = [
        "How does the GET method work with caching?",
        "What happens when this request fails?",
        "How are these headers processed?",
        "What does the Content-Length header do?",
        "How is chunked transfer encoding implemented?",
        "What are the requirements for persistent connections?",
        "How does content negotiation work?",
        "What happens with conditional requests?",
        "How are authentication credentials handled?",
        "What is the difference between this and that method?"
    ]
    
    # Track performance by query type
    results_by_type = {qtype: {"traditional": [], "late": [], "improvements": []} 
                      for qtype in TECHNICAL_QUERY_TYPES.keys()}
    results_by_type["unknown"] = {"traditional": [], "late": [], "improvements": []}
    
    print("=" * 80)
    print("LATE CHUNKING vs TRADITIONAL CHUNKING - TECHNICAL DOCUMENTATION DEMO")
    print("=" * 80)
    print()
    
    # Initialize both chunkers with settings optimized for technical text
    traditional_chunker = TraditionalChunker(chunk_size=200)
    late_chunker = LateChunker(context_window=800, chunk_size=200)
    
    # Process the RFC with both approaches
    traditional_chunker.process_document(rfc_text)
    print()
    late_chunker.process_document(rfc_text)
    print()
    
    # Show context example
    print("TECHNICAL CONTEXT VISUALIZATION")
    print("-" * 50)
    late_chunker.show_context_example(15)
    
    # Run comparison for each test query
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}")
        print()
        
        # Traditional chunking results
        print("TRADITIONAL CHUNKING RESULTS:")
        print("-" * 50)
        traditional_results = traditional_chunker.search(query, top_k=2)
        
        for j, (chunk, score) in enumerate(traditional_results, 1):
            print(f"Result {j} (Score: {score:.4f}):")
            print(f"'{chunk[:300]}{'...' if len(chunk) > 300 else ''}'")
            print()
        
        # Late chunking results
        print("LATE CHUNKING RESULTS:")
        print("-" * 50)
        late_results = late_chunker.search(query, top_k=2)
        
        for j, (chunk, score) in enumerate(late_results, 1):
            print(f"Result {j} (Score: {score:.4f}):")
            print(f"'{chunk[:300]}{'...' if len(chunk) > 300 else ''}'")
            print()
        
        # Compare scores
        traditional_max_score = max(score for _, score in traditional_results)
        late_max_score = max(score for _, score in late_results)
        improvement = ((late_max_score - traditional_max_score) / traditional_max_score * 100)
        
        # Track results by query type
        query_type = classify_technical_query(query)
        results_by_type[query_type]["traditional"].append(traditional_max_score)
        results_by_type[query_type]["late"].append(late_max_score)
        results_by_type[query_type]["improvements"].append(improvement)
        
        print("COMPARISON:")
        print(f"Query type: {query_type}")
        print(f"Traditional max score: {traditional_max_score:.4f}")
        print(f"Late chunking max score: {late_max_score:.4f}")
        print(f"Late chunking improvement: {improvement:+.2f}%")
        print()
    
    # Summary by query type
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS BY TECHNICAL QUERY TYPE")
    print("="*80)
    for qtype, data in results_by_type.items():
        if data["improvements"]:
            avg_improvement = np.mean(data["improvements"])
            print(f"{qtype.upper().replace('_', ' ')} queries: {avg_improvement:+.2f}% average improvement")
            if qtype in TECHNICAL_QUERY_TYPES and len(TECHNICAL_QUERY_TYPES[qtype]) >= 2:
                print(f"  Examples: {TECHNICAL_QUERY_TYPES[qtype][0]}")
                print(f"            {TECHNICAL_QUERY_TYPES[qtype][1]}")
            print()
    
    # Overall summary
    all_improvements = []
    for data in results_by_type.values():
        all_improvements.extend(data["improvements"])
    
    if all_improvements:
        overall_avg = np.mean(all_improvements)
        wins = sum(1 for imp in all_improvements if imp > 0)
        total = len(all_improvements)
        
        print(f"OVERALL AVERAGE IMPROVEMENT: {overall_avg:+.2f}%")
        print(f"LATE CHUNKING WINS: {wins}/{total} queries ({wins/total*100:.1f}%)")
        print()
        
        if overall_avg > 5:
            print("🎉 LATE CHUNKING DOMINATES! Technical documentation with cross-references")
            print("   benefits significantly from preserved context and procedural continuity.")
        elif overall_avg > 0:
            print("✅ LATE CHUNKING WINS! Technical docs show clear advantages with context preservation.")
        else:
            print("📊 Mixed results. Some technical queries may not require extended context.")


if __name__ == "__main__":
    print("Starting Technical Documentation Late Chunking Demo...")
    print("Using RFC 2616 (HTTP/1.1 Specification)")
    print("This demo requires sentence-transformers, numpy, and scikit-learn")
    print()
    
    try:
        run_technical_demo()
        
        print("\n" + "="*80)
        print("TECHNICAL DOCUMENTATION INSIGHTS")
        print("="*80)
        print("""
Why Late Chunking Should Excel with Technical Documentation:

1. CROSS-REFERENCES: Technical docs are full of "this method", "such headers",
   "the above algorithm" that require context to understand properly.

2. PROTOCOL DEFINITIONS: HTTP methods, headers, and status codes are defined
   once but referenced throughout the document.

3. PROCEDURAL FLOWS: Understanding request/response cycles requires knowing
   the steps that come before and after each operation.

4. TECHNICAL DEPENDENCIES: Error handling, authentication, and caching all
   depend on understanding the broader protocol context.

5. FORMAL LANGUAGE: Technical specifications use precise language with lots
   of pronouns and references that need context resolution.

RFC documents represent an ideal use case for late chunking because they're
designed as interconnected specifications where every part references others.
        """)
        
    except FileNotFoundError:
        print("Error: rfc2616.txt not found!")
        print("Please download it first:")
        print("curl -o rfc2616.txt https://www.rfc-editor.org/rfc/rfc2616.txt")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install sentence-transformers numpy scikit-learn")
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure all dependencies are installed and try again.")