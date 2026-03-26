#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clear any existing figures
plt.close('all')

# Load data
df = pd.read_csv('benchmark_results.csv')

# Combine encoding + search into "Document Retrieval"
df['retrieval_ms'] = df['encoding_ms'] + df['search_ms']

# Sample 20 random queries (or all if less than 20)
np.random.seed(42)  # For reproducibility
n_queries = min(20, len(df))
sample_indices = np.random.choice(len(df), n_queries, replace=False)
sample_indices = np.sort(sample_indices)
df_sample = df.iloc[sample_indices].reset_index(drop=True)

# The 3 components
components = ['retrieval_ms', 'augmentation_ms', 'generation_ms']
labels = ['Document Retrieval', 'Question Augmentation', 'LLM Generation']
colors = ['#3498db', '#2ecc71', '#e74c3c']

# Calculate statistics (on full dataset)
stats = {}
for comp, label in zip(components, labels):
    stats[label] = {
        'mean': df[comp].mean(),
        'median': df[comp].median(),
        'std': df[comp].std(),
        'min': df[comp].min(),
        'max': df[comp].max(),
        'pct': (df[comp].mean() / df['total_ms'].mean()) * 100
    }

# Create figure with 4 plots (2x2)
fig = plt.figure(figsize=(16, 12))

query_indices = range(n_queries)

# Plot 1: Document Retrieval distribution (bar per query)
ax1 = fig.add_subplot(2, 2, 1)
bars1 = ax1.bar(query_indices, df_sample['retrieval_ms'], color=colors[0], alpha=0.8, edgecolor='black')
ax1.axhline(df['retrieval_ms'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["retrieval_ms"].mean():.2f}ms')
ax1.set_xlabel('Query Index', fontweight='bold', fontsize=12)
ax1.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
ax1.set_title('Document Retrieval Distribution (20 queries)', fontweight='bold', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_xticks(query_indices)

# Plot 2: Question Augmentation distribution (bar per query)
ax2 = fig.add_subplot(2, 2, 2)
bars2 = ax2.bar(query_indices, df_sample['augmentation_ms'], color=colors[1], alpha=0.8, edgecolor='black')
ax2.axhline(df['augmentation_ms'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["augmentation_ms"].mean():.4f}ms')
ax2.set_xlabel('Query Index', fontweight='bold', fontsize=12)
ax2.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
ax2.set_title('Question Augmentation Distribution (20 queries)', fontweight='bold', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_xticks(query_indices)

# Plot 3: LLM Generation distribution (bar per query)
ax3 = fig.add_subplot(2, 2, 3)
bars3 = ax3.bar(query_indices, df_sample['generation_ms'], color=colors[2], alpha=0.8, edgecolor='black')
ax3.axhline(df['generation_ms'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["generation_ms"].mean():.2f}ms')
ax3.set_xlabel('Query Index', fontweight='bold', fontsize=12)
ax3.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
ax3.set_title('LLM Generation Distribution (20 queries)', fontweight='bold', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_xticks(query_indices)

# Plot 4: Pie chart of percentage breakdown
ax4 = fig.add_subplot(2, 2, 4)
percentages = [stats[l]['pct'] for l in labels]
explode = (0, 0, 0.05)
wedges, texts, autotexts = ax4.pie(percentages, labels=labels, colors=colors, 
                                    autopct='%1.2f%%', explode=explode,
                                    shadow=True, startangle=90)
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
ax4.set_title('Time Breakdown (% of Total)', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('three_step_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: three_step_distribution.png")

# Print analysis
print("\n" + "="*80)
print("RAG PIPELINE LATENCY ANALYSIS - 3 COMPONENTS")
print("="*80)

print("\n1. TIME BREAKDOWN OF COMPONENTS")
print("-"*80)
print(f"{'Component':<25} {'Mean':<12} {'Median':<12} {'Std Dev':<12} {'% of Total':<12}")
print("-"*80)
for label in labels:
    s = stats[label]
    print(f"{label:<25} {s['mean']:>9.2f}ms {s['median']:>9.2f}ms {s['std']:>9.2f}ms {s['pct']:>9.2f}%")
print("-"*80)
total_mean = df['total_ms'].mean()
total_median = df['total_ms'].median()
total_std = df['total_ms'].std()
print(f"{'TOTAL':<25} {total_mean:>9.2f}ms {total_median:>9.2f}ms {total_std:>9.2f}ms {'100.00':>9}%")

print("\n" + "="*80)
print("2. MAJOR BOTTLENECK")
print("="*80)
bottleneck = max(stats.items(), key=lambda x: x[1]['pct'])
print(f"\nBOTTLENECK: {bottleneck[0]}")
print(f"Time: {bottleneck[1]['mean']:.2f}ms ({bottleneck[1]['pct']:.2f}% of total)")
print(f"\nLLM Generation is the DOMINANT bottleneck ({bottleneck[1]['pct']:.2f}% of total time)")
print(f"- Document Retrieval: {stats['Document Retrieval']['pct']:.2f}%")
print(f"- Question Augmentation: {stats['Question Augmentation']['pct']:.2f}%")

print("\n" + "="*80)
print("3. OPTIMIZATION OPPORTUNITIES")
print("="*80)
print("""
PRIORITY 1: LLM Generation (99%+ of time)
  - Use smaller/faster LLM model (TinyLlama vs Qwen2)
  - Reduce max_tokens for shorter responses
  - Use quantized models (4-bit, 8-bit)
  - Enable GPU acceleration
  - Use speculative decoding

PRIORITY 2: Document Retrieval (<1% of time)
  - Use IVFFlat index instead of Flat (approximate search)
  - Batch multiple queries together (FAISS batch_search)
  - Reduce embedding dimension
  - Use faster embedding model

PRIORITY 3: Question Augmentation (negligible)
  - Already optimized (just string concatenation)
  - Reduce Top-K to include fewer documents
""")

