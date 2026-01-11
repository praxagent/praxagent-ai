"""
Late Chunking Implementation and Analysis

This script demonstrates late chunking vs traditional chunking for text embeddings,
comparing performance across various test terms and visualizing the results.

Late chunking preserves context during embedding generation, then pools at the chunk level,
while traditional chunking embeds each chunk independently.
"""

# Standard library imports
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Third-party imports
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

# ML/NLP imports
from transformers import AutoModel, AutoTokenizer


def cosine_similarity(x, y):
    """Calculate cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def load_model_and_tokenizer():
    """Load the embedding model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    return model, tokenizer

def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def chunk_by_tokenizer_api(input_text: str, tokenizer: callable):
    """Alternative chunking method using Jina AI Tokenizer API."""
    url = 'https://tokenize.jina.ai/'
    payload = {
        "content": input_text,
        "return_chunks": "true",
        "max_chunk_length": "1000"
    }

    response = requests.post(url, json=payload)
    response_data = response.json()

    chunks = response_data.get("chunks", [])
    chunk_positions = response_data.get("chunk_positions", [])
    span_annotations = [(start, end) for start, end in chunk_positions]

    return chunks, span_annotations

def late_chunking(model_output, span_annotation: list, max_length=None):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs



def comprehensive_chunking_analysis(model, chunks, embeddings, embeddings_traditional_chunking):
    """
    Comprehensive analysis of late chunking vs traditional chunking across multiple terms
    """

    # Define diverse test terms with expected relatedness to chunks
    test_terms = {
        # Cities (Berlin-related and unrelated)
        'cities_related': {
            'terms': ['Berlin', 'Brandenburg', 'Germany', 'Deutsche', 'European capital', 'German capital', 
                     'Federal Republic', 'German state', 'European Union', 'Central Europe', 'Germanic',
                     'Deutschland', 'Prussian', 'Habsburg', 'Rhine', 'Bavaria', 'Munich relation'],
            'expected_relatedness': 'high'  # Should have HIGH similarity to Berlin-focused chunks
        },


        # Animals - should be UNRELATED to Berlin/city chunks
        'animals': {
            'terms': ['dog', 'cat', 'elephant', 'tiger', 'lion', 'bear', 'wolf', 'rabbit'],
            'expected_relatedness': 'low'
        },

        # Technology - might have some relation if chunks mention tech
        'technology': {
            'terms': ['computer', 'smartphone', 'artificial intelligence', 'machine learning', 'software', 'algorithm'],
            'expected_relatedness': 'low'
        },

        # Food - should be UNRELATED
        'food': {
            'terms': ['pizza', 'sushi', 'pasta', 'hamburger', 'salad', 'chocolate'],
            'expected_relatedness': 'low'
        },

        # Nature - should be UNRELATED
        'nature': {
            'terms': ['mountain', 'ocean', 'forest', 'river', 'flower', 'sunset'],
            'expected_relatedness': 'low'
        },

        # Abstract concepts - should be UNRELATED
        'abstract': {
            'terms': ['democracy', 'freedom', 'justice', 'creativity', 'innovation', 'happiness'],
            'expected_relatedness': 'low'
        },



        # Historical - might have some relation if chunks mention history
        'historical': {
            'terms': ['Napoleon', 'Renaissance', 'Industrial Revolution', 'Cold War', 'Ancient Rome'],
            'expected_relatedness': 'low'
        },

        # Random/Unrelated - should be UNRELATED
        'random': {
            'terms': ['bicycle', 'purple', 'mathematics', 'music', 'basketball', 'astronomy'],
            'expected_relatedness': 'low'
        }
    }

    # Flatten all terms for easy iteration
    all_terms = []
    term_categories = {}
    term_expected_relatedness = {}
    for category, data in test_terms.items():
        terms = data['terms']
        expected = data['expected_relatedness']
        for term in terms:
            all_terms.append(term)
            term_categories[term] = category
            term_expected_relatedness[term] = expected

    print("=" * 80)
    print("COMPREHENSIVE LATE CHUNKING vs TRADITIONAL CHUNKING ANALYSIS")
    print("=" * 80)
    print(f"Testing {len(all_terms)} terms across {len(test_terms)} categories")
    print(f"Analyzing {len(chunks)} chunks\n")

    # Generate embeddings for all test terms
    term_embeddings = {}
    print("Generating embeddings for test terms...")
    for term in all_terms:
        term_embeddings[term] = model.encode(term)

    # Store results for analysis
    results = []
    detailed_results = defaultdict(list)

    # Define what constitutes "good performance" based on expected relatedness
    def calculate_directional_improvement(late_sim, trad_sim, expected_relatedness):
        """
        Calculate improvement considering the expected direction:
        - For HIGH expected relatedness: Higher similarity is better (we want to find related content)
        - For LOW expected relatedness: Lower similarity is better (we want to avoid false matches)
        """
        raw_improvement = late_sim - trad_sim

        if expected_relatedness == 'high':
            # For related terms, positive improvement is good
            return raw_improvement, raw_improvement > 0
        elif expected_relatedness == 'low':
            # For unrelated terms, negative improvement is actually good (lower false similarity)
            return -raw_improvement, raw_improvement < 0  # Flip the sign
        else:
            # Should not reach here since we only have high and low now
            raise ValueError(f"Unexpected relatedness level: {expected_relatedness}")

    # Analyze each term against each chunk
    for term in all_terms:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {term.upper()} ({term_categories[term]})")
        print(f"{'='*60}")

        term_embedding = term_embeddings[term]
        term_results = {'late': [], 'traditional': []}

        for i, (chunk, new_embedding, trad_embedding) in enumerate(zip(chunks, embeddings, embeddings_traditional_chunking)):
            late_sim = cosine_similarity(term_embedding, new_embedding)
            trad_sim = cosine_similarity(term_embedding, trad_embedding)
            raw_difference = late_sim - trad_sim
            expected_rel = term_expected_relatedness[term]

            # Calculate directional improvement
            directional_improvement, is_directional_better = calculate_directional_improvement(
                late_sim, trad_sim, expected_rel
            )

            # Store for statistical analysis
            term_results['late'].append(late_sim)
            term_results['traditional'].append(trad_sim)

            results.append({
                'term': term,
                'category': term_categories[term],
                'expected_relatedness': expected_rel,
                'chunk_id': i,
                'chunk_preview': chunk[:50] + "..." if len(chunk) > 50 else chunk,
                'late_similarity': late_sim,
                'traditional_similarity': trad_sim,
                'raw_difference': raw_difference,
                'directional_improvement': directional_improvement,
                'late_better_raw': late_sim > trad_sim,
                'late_better_directional': is_directional_better
            })

            # Print detailed comparison with directional context
            print(f'\nChunk {i+1}: "{chunk[:60]}{"..." if len(chunk) > 60 else ""}"')
            print(f'  Late Chunking:           {late_sim:.4f}')
            print(f'  Traditional:             {trad_sim:.4f}')
            print(f'  Raw Difference:          {raw_difference:+.4f}')
            print(f'  Expected Relatedness:    {expected_rel.upper()}')
            print(f'  Directional Improvement: {directional_improvement:+.4f} {"✓" if is_directional_better else "✗"}')

            # Add interpretation
            if expected_rel == 'high':
                print(f'    → We WANT high similarity here, so {"higher" if raw_difference > 0 else "lower"} is {"good" if raw_difference > 0 else "bad"}')
            elif expected_rel == 'low':
                print(f'    → We WANT low similarity here, so {"lower" if raw_difference < 0 else "higher"} is {"good" if raw_difference < 0 else "bad"}')
            else:
                print(f'    → We WANT moderate similarity here (~0.45)')

        # Calculate statistics for this term
        avg_late = np.mean(term_results['late'])
        avg_trad = np.mean(term_results['traditional'])
        std_late = np.std(term_results['late'])
        std_trad = np.std(term_results['traditional'])

        # Calculate directional statistics
        term_directional_improvements = [r['directional_improvement'] for r in results if r['term'] == term]
        avg_directional_improvement = np.mean(term_directional_improvements)
        directional_wins = sum(1 for r in results if r['term'] == term and r['late_better_directional'])

        detailed_results[term] = {
            'category': term_categories[term],
            'expected_relatedness': expected_rel,
            'avg_late': avg_late,
            'avg_traditional': avg_trad,
            'std_late': std_late,
            'std_traditional': std_trad,
            'raw_improvement': avg_late - avg_trad,
            'directional_improvement': avg_directional_improvement,
            'late_wins_raw': sum(1 for r in results if r['term'] == term and r['late_better_raw']),
            'late_wins_directional': directional_wins,
            'total_chunks': len(chunks)
        }

        print(f'\n📊 SUMMARY for {term} (Expected: {expected_rel.upper()} relatedness):')
        print(f'  Average Late:              {avg_late:.4f} (±{std_late:.4f})')
        print(f'  Average Traditional:       {avg_trad:.4f} (±{std_trad:.4f})')
        print(f'  Raw Improvement:           {avg_late - avg_trad:+.4f}')
        print(f'  Directional Improvement:   {avg_directional_improvement:+.4f}')
        print(f'  Raw Wins:                  {detailed_results[term]["late_wins_raw"]}/{len(chunks)}')
        print(f'  Directional Wins:          {directional_wins}/{len(chunks)} ⭐')

        # Interpretation
        if expected_rel == 'high' and avg_late > avg_trad:
            print(f'  🎯 GOOD: Late chunking found more relevant content!')
        elif expected_rel == 'low' and avg_late < avg_trad:
            print(f'  🎯 GOOD: Late chunking avoided false matches!')
        elif expected_rel == 'low' and avg_late > avg_trad:
            print(f'  ⚠️  CONCERNING: Late chunking may be creating false matches!')
        elif expected_rel == 'high' and avg_late < avg_trad:
            print(f'  ⚠️  CONCERNING: Late chunking missed relevant content!')

    # Create comprehensive visualizations
    create_comprehensive_plots(results, detailed_results, test_terms)

    # Generate final summary statistics
    generate_final_summary(detailed_results, test_terms)

    return results, detailed_results

def create_related_vs_unrelated_comparison(df, save_dir):
    """Create side-by-side comparison plots for related vs unrelated terms"""
    
    # Separate data by expected relatedness
    high_rel_data = df[df['expected_relatedness'] == 'high']
    low_rel_data = df[df['expected_relatedness'] == 'low']
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: HIGH RELATEDNESS (should favor late chunking)
    if not high_rel_data.empty:
        high_late_avg = high_rel_data.groupby('term')['late_similarity'].mean()
        high_trad_avg = high_rel_data.groupby('term')['traditional_similarity'].mean()
        
        x = range(len(high_late_avg))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], high_late_avg.values, width, 
                label='Late Chunking', color='#2E86AB', alpha=0.8)
        ax1.bar([i + width/2 for i in x], high_trad_avg.values, width,
                label='Traditional', color='#A23B72', alpha=0.8)
        
        ax1.set_xlabel('Related Terms (Berlin, Germany, etc.)')
        ax1.set_ylabel('Average Similarity Score')
        ax1.set_title('HIGH RELATEDNESS: Late Chunking Should Win\n(Higher bars = Better)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(high_late_avg.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
    
    # Plot 2: LOW RELATEDNESS (should favor traditional chunking or lower scores)
    if not low_rel_data.empty:
        # Sample a subset of low relatedness terms for readability
        low_sample = low_rel_data.groupby('term').first().sample(n=min(10, len(low_rel_data.groupby('term'))), random_state=42)
        low_late_avg = low_rel_data[low_rel_data['term'].isin(low_sample.index)].groupby('term')['late_similarity'].mean()
        low_trad_avg = low_rel_data[low_rel_data['term'].isin(low_sample.index)].groupby('term')['traditional_similarity'].mean()
        
        x = range(len(low_late_avg))
        
        ax2.bar([i - width/2 for i in x], low_late_avg.values, width,
                label='Late Chunking', color='#2E86AB', alpha=0.8)
        ax2.bar([i + width/2 for i in x], low_trad_avg.values, width,
                label='Traditional', color='#A23B72', alpha=0.8)
        
        ax2.set_xlabel('Unrelated Terms (Animals, Food, etc.)')
        ax2.set_ylabel('Average Similarity Score')
        ax2.set_title('LOW RELATEDNESS: Lower Scores Better\n(Lower bars = Better, prevents false matches)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(low_late_avg.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'related_vs_unrelated_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_improved_scatter_plot(detailed_results, save_dir):
    """Create an improved scatter plot with better colors and clearer categories"""
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    terms = list(detailed_results.keys())
    late_avgs = [detailed_results[term]['avg_late'] for term in terms]
    trad_avgs = [detailed_results[term]['avg_traditional'] for term in terms]
    categories = [detailed_results[term]['category'] for term in terms]
    expected_rel = [detailed_results[term]['expected_relatedness'] for term in terms]
    
    # Define better colors and markers by expected relatedness
    color_map = {
        'high': '#D32F2F',     # Red for high relatedness
        'medium': '#FF9800',   # Orange for medium
        'low': '#1976D2'       # Blue for low relatedness
    }
    
    marker_map = {
        'high': 'o',      # Circle
        'medium': 's',    # Square  
        'low': '^'        # Triangle
    }
    
    # Plot points grouped by expected relatedness
    for rel_level in ['high', 'medium', 'low']:
        mask = [r == rel_level for r in expected_rel]
        if any(mask):
            late_subset = [late_avgs[i] for i in range(len(late_avgs)) if mask[i]]
            trad_subset = [trad_avgs[i] for i in range(len(trad_avgs)) if mask[i]]
            terms_subset = [terms[i] for i in range(len(terms)) if mask[i]]
            
            scatter = plt.scatter(trad_subset, late_subset, 
                                c=color_map[rel_level], 
                                marker=marker_map[rel_level],
                                s=120, alpha=0.7, edgecolors='black', linewidth=1,
                                label=f'{rel_level.title()} Relatedness')
            
            # Add text labels for high relatedness terms
            if rel_level == 'high':
                for i, term in enumerate(terms_subset):
                    plt.annotate(term, (trad_subset[i], late_subset[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, alpha=0.8)
    
    # Diagonal line (equal performance)
    max_val = max(max(late_avgs), max(trad_avgs))
    min_val = min(min(late_avgs), min(trad_avgs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, 
             label='Equal Performance (y=x)')
    
    # Add linear regression line to show actual relationship
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(trad_avgs, late_avgs)
    
    # Create regression line
    regression_x = np.linspace(min_val, max_val, 100)
    regression_y = slope * regression_x + intercept
    plt.plot(regression_x, regression_y, 'r-', alpha=0.8, linewidth=3,
             label=f'Best Fit: y={slope:.3f}x+{intercept:.3f} (R²={r_value**2:.3f})')
    
    # Add text box with regression statistics
    textstr = f'Best Fit Analysis:\n'
    textstr += f'Slope: {slope:.3f}\n'
    textstr += f'R²: {r_value**2:.3f}\n'
    if slope > 1:
        textstr += 'Late chunking amplifies differences'
    elif slope < 1:
        textstr += 'Late chunking moderates differences'
    else:
        textstr += 'Late chunking maintains proportions'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    

    
    plt.xlabel('Traditional Chunking Similarity', fontsize=12)
    plt.ylabel('Late Chunking Similarity', fontsize=12)
    plt.title('Late vs Traditional Performance by Term Relatedness\n(RED points: above diagonal = Late wins | BLUE points: below diagonal = Late wins)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improved_performance_scatter.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comprehensive_plots(results, detailed_results, test_terms, save_dir='demo_figures'):
    """Create comprehensive visualizations of the results and save as separate PNG files"""
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create separate plots for related vs unrelated terms
    create_related_vs_unrelated_comparison(df, save_dir)
    create_improved_scatter_plot(detailed_results, save_dir)
    
    # Plot 1: Removed - similarity by category

    # Plot 2: Directional improvement distribution
    plt.figure(figsize=(12, 8))
    raw_improvements = [r['raw_difference'] for r in results]
    directional_improvements = [r['directional_improvement'] for r in results]

    plt.hist(raw_improvements, bins=20, alpha=0.5, color='blue', label='Raw Improvement', edgecolor='black')
    plt.hist(directional_improvements, bins=20, alpha=0.5, color='green', label='Directional Improvement', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Improvement Score')
    plt.ylabel('Number of Comparisons')
    plt.title('Raw vs Directional Improvements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improvement_distribution.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Plot 3: Removed - replaced by improved_performance_scatter.png

    # Plot 4: Directional win rate by expected relatedness (excluding medium)
    plt.figure(figsize=(12, 8))
    relatedness_levels = ['high', 'low']  # Removed medium due to calculation issues
    directional_win_rates = []

    for rel_level in relatedness_levels:
        rel_results = [r for r in results if r['expected_relatedness'] == rel_level]
        if rel_results:
            directional_wins = sum(1 for r in rel_results if r['late_better_directional'])
            total = len(rel_results)
            directional_win_rates.append(directional_wins / total)
        else:
            directional_win_rates.append(0)

    x = np.arange(len(relatedness_levels))
    
    # Use different colors for high vs low
    colors = ['#D32F2F', '#1976D2']  # Red for high, blue for low
    bars = plt.bar(x, directional_win_rates, width=0.5, alpha=0.8, 
                   color=colors, edgecolor='black', linewidth=1)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                label='50% Baseline')
    plt.ylabel('Directional Win Rate', fontsize=12)
    plt.title('Late Chunking Directional Performance\n(High vs Low Relatedness Only)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, [f'{rel.title()} Relatedness' for rel in relatedness_levels])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'win_rates_by_relatedness.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Plot 5: Removed - similarity distribution boxplot

    # Plot 6: Heatmap grouped by relatedness levels (stacked subplots)
    
    # Group terms by relatedness
    relatedness_groups = {'high': [], 'medium': [], 'low': []}
    for term, stats in detailed_results.items():
        rel_level = stats['expected_relatedness']
        relatedness_groups[rel_level].append(term)
    
    # Calculate smart color range based on actual data variation
    all_similarities = []
    for term, stats in detailed_results.items():
        all_similarities.extend([stats['avg_late'], stats['avg_traditional']])
    
    vmin = min(all_similarities)
    vmax = max(all_similarities)
    # Add some padding to make differences more visible
    range_padding = (vmax - vmin) * 0.1
    vmin = max(0, vmin - range_padding)
    vmax = min(1, vmax + range_padding)
    
    # Only create subplots for existing relatedness levels
    existing_levels = [(level, label, color) for level, label, color in 
                      zip(['high', 'low'], ['HIGH Relatedness', 'LOW Relatedness'], ['#D32F2F', '#1976D2'])
                      if relatedness_groups[level]]
    
    fig, axes = plt.subplots(len(existing_levels), 1, figsize=(14, 6 * len(existing_levels)))
    
    # Handle case where we only have one subplot
    if len(existing_levels) == 1:
        axes = [axes]
    
    for i, ((rel_level, group_label, color), ax) in enumerate(zip(existing_levels, axes)):
        terms_in_group = relatedness_groups[rel_level]
        
        # Get similarity data for this group
        late_avgs = [detailed_results[term]['avg_late'] for term in terms_in_group]
        trad_avgs = [detailed_results[term]['avg_traditional'] for term in terms_in_group]
        
        heatmap_data = np.array([late_avgs, trad_avgs])
        
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', 
                      vmin=vmin, vmax=vmax)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Late', 'Traditional'], fontsize=11)
        ax.set_xticks(range(len(terms_in_group)))
        ax.set_xticklabels(terms_in_group, rotation=45, ha='right', fontsize=9)
        
        # Add group title with color coding
        ax.set_title(f'{group_label} Terms', fontsize=13, fontweight='bold', 
                    color=color, pad=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='x')
    
    # Add overall title
    fig.suptitle('Similarity Heatmaps by Term Relatedness', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Similarity Score', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.93)
    plt.savefig(os.path.join(save_dir, 'similarity_heatmap_by_term.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"All plots saved to {save_dir}/ directory:")
    print(f"  🎯 NEW IMPROVED PLOTS:")
    print(f"    - related_vs_unrelated_comparison.png (KEY INSIGHT PLOT)")
    print(f"    - improved_performance_scatter.png (BETTER COLORS & CLARITY)")
    print(f"  📊 REMAINING PLOTS:")
    print(f"    - improvement_distribution.png") 
    print(f"    - win_rates_by_relatedness.png")
    print(f"    - similarity_heatmap_by_term.png")

def generate_final_summary(detailed_results, test_terms):
    """Generate comprehensive final summary statistics"""

    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("="*80)

    # Overall statistics
    all_raw_improvements = [stats['raw_improvement'] for stats in detailed_results.values()]
    all_directional_improvements = [stats['directional_improvement'] for stats in detailed_results.values()]
    all_raw_wins = [stats['late_wins_raw'] for stats in detailed_results.values()]
    all_directional_wins = [stats['late_wins_directional'] for stats in detailed_results.values()]
    all_total_chunks = [stats['total_chunks'] for stats in detailed_results.values()]

    overall_raw_improvement = np.mean(all_raw_improvements)
    overall_directional_improvement = np.mean(all_directional_improvements)
    total_raw_wins = sum(all_raw_wins)
    total_directional_wins = sum(all_directional_wins)
    total_comparisons = sum(all_total_chunks)
    raw_win_rate = total_raw_wins / total_comparisons
    directional_win_rate = total_directional_wins / total_comparisons

    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"  Raw Improvement:         {overall_raw_improvement:+.4f}")
    print(f"  Directional Improvement: {overall_directional_improvement:+.4f} ⭐")
    print(f"  Raw Win Rate:            {raw_win_rate:.1%} ({total_raw_wins}/{total_comparisons})")
    print(f"  Directional Win Rate:    {directional_win_rate:.1%} ({total_directional_wins}/{total_comparisons}) ⭐")
    print(f"  Std Dev (Raw):           {np.std(all_raw_improvements):.4f}")
    print(f"  Std Dev (Directional):   {np.std(all_directional_improvements):.4f}")

    # Expected relatedness breakdown
    print(f"\n📊 PERFORMANCE BY EXPECTED RELATEDNESS:")
    relatedness_levels = ['high', 'medium', 'low']
    for rel_level in relatedness_levels:
        rel_terms = [term for term, stats in detailed_results.items()
                    if stats['expected_relatedness'] == rel_level]
        if rel_terms:
            rel_raw_improvements = [detailed_results[term]['raw_improvement'] for term in rel_terms]
            rel_directional_improvements = [detailed_results[term]['directional_improvement'] for term in rel_terms]
            rel_directional_wins = [detailed_results[term]['late_wins_directional'] for term in rel_terms]
            rel_total_chunks = [detailed_results[term]['total_chunks'] for term in rel_terms]

            avg_raw = np.mean(rel_raw_improvements)
            avg_directional = np.mean(rel_directional_improvements)
            directional_win_rate = sum(rel_directional_wins) / sum(rel_total_chunks)

            print(f"  {rel_level.upper():6} Relatedness | Raw: {avg_raw:+.4f} | Directional: {avg_directional:+.4f} | Win Rate: {directional_win_rate:.1%}")

    # Category breakdown
    print(f"\n📊 PERFORMANCE BY CATEGORY:")
    for category in test_terms.keys():
        category_terms = [term for term, stats in detailed_results.items()
                         if stats['category'] == category]
        if category_terms:
            cat_directional_improvements = [detailed_results[term]['directional_improvement'] for term in category_terms]
            cat_directional_wins = [detailed_results[term]['late_wins_directional'] for term in category_terms]
            cat_total_chunks = [detailed_results[term]['total_chunks'] for term in category_terms]

            cat_avg_directional = np.mean(cat_directional_improvements)
            cat_directional_win_rate = sum(cat_directional_wins) / sum(cat_total_chunks)

            print(f"  {category:15} | Directional Improvement: {cat_avg_directional:+.4f} | Win Rate: {cat_directional_win_rate:.1%}")

    # Top and bottom performers (by directional improvement)
    sorted_by_directional = sorted(detailed_results.items(),
                                 key=lambda x: x[1]['directional_improvement'], reverse=True)

    print(f"\n🏆 TOP 5 TERMS (Best Directional Performance):")
    for i, (term, stats) in enumerate(sorted_by_directional[:5]):
        expected = stats['expected_relatedness']
        directional = stats['directional_improvement']
        print(f"  {i+1:2}. {term:20} | {directional:+.4f} | Expected: {expected:6} | {stats['category']}")

    print(f"\n📉 BOTTOM 5 TERMS (Worst Directional Performance):")
    for i, (term, stats) in enumerate(sorted_by_directional[-5:]):
        expected = stats['expected_relatedness']
        directional = stats['directional_improvement']
        print(f"  {i+1:2}. {term:20} | {directional:+.4f} | Expected: {expected:6} | {stats['category']}")

    # Statistical significance test (using previously imported scipy_stats)

    # Test both raw and directional improvements
    raw_similarities_late = [stats['avg_late'] for stats in detailed_results.values()]
    raw_similarities_trad = [stats['avg_traditional'] for stats in detailed_results.values()]
    directional_improvements_only = [stats['directional_improvement'] for stats in detailed_results.values()]

    t_stat_raw, p_value_raw = scipy_stats.ttest_rel(raw_similarities_late, raw_similarities_trad)
    t_stat_directional, p_value_directional = scipy_stats.ttest_1samp(directional_improvements_only, 0)

    print(f"\n📈 STATISTICAL ANALYSIS:")
    print(f"  Raw Similarities Paired t-test:")
    print(f"    t-statistic: {t_stat_raw:.4f}")
    print(f"    p-value:     {p_value_raw:.6f}")
    print(f"    Significant: {'Yes' if p_value_raw < 0.05 else 'No'} (α = 0.05)")

    print(f"  Directional Improvements One-sample t-test (vs 0):")
    print(f"    t-statistic: {t_stat_directional:.4f}")
    print(f"    p-value:     {p_value_directional:.6f}")
    print(f"    Significant: {'Yes' if p_value_directional < 0.05 else 'No'} (α = 0.05)")

    # Conclusion
    print(f"\n🎯 CONCLUSION:")
    print(f"  📊 RAW ANALYSIS:")
    if overall_raw_improvement > 0 and p_value_raw < 0.05:
        print("    ✅ Late Chunking shows statistically significant RAW improvement!")
    elif overall_raw_improvement > 0:
        print("    ⚠️  Late Chunking shows RAW improvement, but not statistically significant.")
    else:
        print("    ❌ Traditional Chunking performs better in RAW terms.")

    print(f"  🎯 DIRECTIONAL ANALYSIS (The Smart Way):")
    if overall_directional_improvement > 0 and p_value_directional < 0.05:
        print("    ✅ Late Chunking shows statistically significant DIRECTIONAL improvement!")
        print("    📈 This means it's better at finding relevant content AND avoiding false matches!")
    elif overall_directional_improvement > 0:
        print("    ⚠️  Late Chunking shows directional improvement, but not statistically significant.")
    else:
        print("    ❌ Traditional Chunking performs better directionally.")
        print("    📉 Late chunking may be creating more false matches or missing relevant content!")

    print(f"\n  📊 SUMMARY:")
    print(f"    Raw improvement:         {overall_raw_improvement:+.4f} ({raw_win_rate:.1%} win rate)")
    print(f"    Directional improvement: {overall_directional_improvement:+.4f} ({directional_win_rate:.1%} win rate) ⭐")
    print("="*80)

def main():
    """Main function to run the late chunking analysis."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Sample input text
    input_text = ("Berlin is the capital and largest city of Germany, both by area and by population. "
                 "Its more than 3.85 million inhabitants make it the European Union's most populous city, "
                 "as measured by population within city limits. The city is also one of the states of Germany, "
                 "and is the third smallest state in the country in terms of area.")
    
    # Determine chunks
    chunks, span_annotations = chunk_by_sentences(input_text, tokenizer)
    print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')
    
    # Traditional chunking (embed each chunk separately)
    embeddings_traditional_chunking = model.encode(chunks)
    
    # Late chunking (embed full context, then pool by chunks)
    inputs = tokenizer(input_text, return_tensors='pt')
    model_output = model(**inputs)
    embeddings = late_chunking(model_output, [span_annotations])[0]
    
    # Comprehensive analysis
    results, detailed_results = comprehensive_chunking_analysis(
        model, chunks, embeddings, embeddings_traditional_chunking
    )
    
    print("\nAnalysis complete. Check the demo_figures directory for visualization outputs.")
    
    return results, detailed_results


if __name__ == "__main__":
    main()