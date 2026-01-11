"""
Late Chunking vs Traditional Chunking Demo - Literary Text
==========================================================

This demo uses Pride and Prejudice by Jane Austen to demonstrate where late chunking
significantly outperforms traditional chunking. Literary texts are rich with pronouns,
character relationships, and contextual dependencies that benefit from preserved context.

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


def clean_gutenberg_text(text: str) -> str:
    """Clean Project Gutenberg text by removing headers/footers and normalizing."""
    lines = text.split('\n')
    
    # Find start of actual content (after "START OF THE PROJECT GUTENBERG")
    start_idx = 0
    for i, line in enumerate(lines):
        if "START OF" in line and "PROJECT GUTENBERG" in line:
            start_idx = i + 1
            break
    
    # Find end of actual content (before "END OF THE PROJECT GUTENBERG")
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if "END OF" in line and "PROJECT GUTENBERG" in line:
            end_idx = i
            break
    
    # Extract main content
    content_lines = lines[start_idx:end_idx]
    content = '\n'.join(content_lines)
    
    # Clean up excessive whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Normalize paragraph breaks
    content = re.sub(r'[ \t]+', ' ', content)      # Normalize spaces
    content = content.strip()
    
    return content


# Literary text query categories that benefit from context
LITERARY_QUERY_TYPES = {
    "character_relationships": [
        "What does Elizabeth think of Mr. Darcy?",
        "How does Darcy feel about Elizabeth?",
        "What is Jane's relationship with Bingley?"
    ],
    "pronoun_resolution": [
        "What does he propose to her?",
        "Why does she reject him?", 
        "How does he react to this?"
    ],
    "plot_development": [
        "What happens at the ball?",
        "Why does Elizabeth change her opinion?",
        "What causes their misunderstanding?"
    ],
    "social_context": [
        "What are the marriage expectations?",
        "How does social class affect relationships?",
        "What role does reputation play?"
    ]
}


def classify_literary_query(query: str) -> str:
    """Classify a literary query by type."""
    for query_type, examples in LITERARY_QUERY_TYPES.items():
        for example in examples:
            if any(word in query.lower() for word in example.lower().split()[:3]):
                return query_type
    return "unknown"


class TraditionalChunker:
    """Traditional chunking approach for literary texts."""
    
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
        cleaned_text = clean_gutenberg_text(text)
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
    """Late chunking approach that preserves literary context."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 context_window: int = 600, chunk_size: int = 200):
        self.model = SentenceTransformer(model_name)
        self.context_window = context_window
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = []
        self.context_chunks = []
    
    def create_contextual_chunks(self, text: str) -> List[Tuple[str, str]]:
        """Create chunks with extended literary context for better character/plot understanding."""
        words = text.split()
        chunks_with_context = []
        
        for i in range(0, len(words), self.chunk_size):
            # Get the main chunk
            chunk_end = min(i + self.chunk_size, len(words))
            chunk = ' '.join(words[i:chunk_end])
            
            # Get extended context for literary understanding
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
        cleaned_text = clean_gutenberg_text(text)
        self.context_chunks = self.create_contextual_chunks(cleaned_text)
        self.chunks = [chunk for chunk, _ in self.context_chunks]
        
        print(f"Created {len(self.chunks)} chunks with extended literary context")
        
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
        """Show how extended context helps with literary understanding."""
        if 0 <= chunk_index < len(self.chunks):
            chunk, context = self.context_chunks[chunk_index]
            print(f"Literary Context Example - Chunk {chunk_index + 1}:")
            print(f"- Core chunk: {len(chunk.split())} words")
            print(f"- Full context: {len(context.split())} words")
            print(f"- Additional context: +{len(context.split()) - len(chunk.split())} words")
            print(f"\nCore chunk: {chunk[:200]}...")
            print(f"\nFull context: {context[:400]}...")
            print()


def run_literary_demo():
    """Run comparison demo using Pride and Prejudice."""
    
    # Load Pride and Prejudice
    print("Loading Pride and Prejudice by Jane Austen...")
    novel_text = load_document("pride_and_prejudice.txt")
    
    # Literary queries that should strongly favor late chunking
    test_queries = [
        "What does Elizabeth think of Mr. Darcy initially?",
        "Why does she reject his proposal?",
        "How does he react to her refusal?",
        "What changes Elizabeth's opinion of him?",
        "What happens at the Netherfield ball?",
        "How do Jane and Bingley meet?",
        "What role does Wickham play in the story?",
        "Why is Elizabeth prejudiced against Darcy?"
    ]
    
    # Track performance by query type
    results_by_type = {qtype: {"traditional": [], "late": [], "improvements": []} 
                      for qtype in LITERARY_QUERY_TYPES.keys()}
    results_by_type["unknown"] = {"traditional": [], "late": [], "improvements": []}
    
    print("=" * 70)
    print("LATE CHUNKING vs TRADITIONAL CHUNKING - LITERARY TEXT DEMO")
    print("=" * 70)
    print()
    
    # Initialize both chunkers with settings optimized for literary text
    traditional_chunker = TraditionalChunker(chunk_size=150)
    late_chunker = LateChunker(context_window=500, chunk_size=150)
    
    # Process the novel with both approaches
    traditional_chunker.process_document(novel_text)
    print()
    late_chunker.process_document(novel_text)
    print()
    
    # Show context example
    print("LITERARY CONTEXT VISUALIZATION")
    print("-" * 40)
    late_chunker.show_context_example(10)
    
    # Run comparison for each test query
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        print()
        
        # Traditional chunking results
        print("TRADITIONAL CHUNKING RESULTS:")
        print("-" * 40)
        traditional_results = traditional_chunker.search(query, top_k=2)
        
        for j, (chunk, score) in enumerate(traditional_results, 1):
            print(f"Result {j} (Score: {score:.4f}):")
            print(f"'{chunk[:250]}{'...' if len(chunk) > 250 else ''}'")
            print()
        
        # Late chunking results
        print("LATE CHUNKING RESULTS:")
        print("-" * 40)
        late_results = late_chunker.search(query, top_k=2)
        
        for j, (chunk, score) in enumerate(late_results, 1):
            print(f"Result {j} (Score: {score:.4f}):")
            print(f"'{chunk[:250]}{'...' if len(chunk) > 250 else ''}'")
            print()
        
        # Compare scores
        traditional_max_score = max(score for _, score in traditional_results)
        late_max_score = max(score for _, score in late_results)
        improvement = ((late_max_score - traditional_max_score) / traditional_max_score * 100)
        
        # Track results by query type
        query_type = classify_literary_query(query)
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
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS BY LITERARY QUERY TYPE")
    print("="*70)
    for qtype, data in results_by_type.items():
        if data["improvements"]:
            avg_improvement = np.mean(data["improvements"])
            print(f"{qtype.upper().replace('_', ' ')} queries: {avg_improvement:+.2f}% average improvement")
            if qtype in LITERARY_QUERY_TYPES and len(LITERARY_QUERY_TYPES[qtype]) >= 2:
                print(f"  Examples: {LITERARY_QUERY_TYPES[qtype][0]}")
                print(f"            {LITERARY_QUERY_TYPES[qtype][1]}")
            print()
    
    # Overall summary
    all_improvements = []
    for data in results_by_type.values():
        all_improvements.extend(data["improvements"])
    
    if all_improvements:
        overall_avg = np.mean(all_improvements)
        print(f"OVERALL AVERAGE IMPROVEMENT: {overall_avg:+.2f}%")
        print()
        
        if overall_avg > 0:
            print("🎉 LATE CHUNKING WINS! Literary texts with rich context benefit significantly")
            print("   from preserved character relationships and narrative continuity.")
        else:
            print("📊 Mixed results. Some literary queries may not require extended context.")


if __name__ == "__main__":
    print("Starting Literary Late Chunking Demo...")
    print("Using Pride and Prejudice by Jane Austen from Project Gutenberg")
    print("This demo requires sentence-transformers, numpy, and scikit-learn")
    print()
    
    try:
        run_literary_demo()
        
        print("\n" + "="*70)
        print("LITERARY TEXT INSIGHTS")
        print("="*70)
        print("""
Why Late Chunking Should Excel with Literary Texts:

1. CHARACTER RELATIONSHIPS: Understanding who "he" and "she" refer to requires
   knowing the characters present in the scene and their relationships.

2. PRONOUN RESOLUTION: Literary texts are full of pronouns that need context
   to resolve correctly - "his proposal", "her refusal", "their dance".

3. EMOTIONAL CONTEXT: Characters' feelings and motivations develop over time
   and require understanding of previous interactions and social dynamics.

4. PLOT CONTINUITY: Events reference earlier scenes, character development,
   and ongoing narrative threads that span multiple paragraphs.

5. SOCIAL NUANCE: Understanding social class, marriage expectations, and
   reputation requires broader context about the characters' world.

Literary texts represent the ideal use case for late chunking because they're
designed to be read as continuous narratives where context enriches meaning.
        """)
        
    except FileNotFoundError:
        print("Error: pride_and_prejudice.txt not found!")
        print("Please download it first:")
        print("curl -o pride_and_prejudice.txt https://www.gutenberg.org/files/1342/1342-0.txt")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install sentence-transformers numpy scikit-learn")
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure all dependencies are installed and try again.")