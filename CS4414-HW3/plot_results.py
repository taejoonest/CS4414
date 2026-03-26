#!/usr/bin/env python3
"""
Visualization script for Part 3 benchmarking results
Generates plots for:
1. Component timing breakdown
2. Top-K comparison
3. Batch size performance
4. Index type comparison (Flat vs IVFFlat)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

plt.style.use('seaborn-v0_8-darkgrid')

def plot_component_breakdown():
    """Plot latency breakdown by component for different top-K values"""
    print("Generating component breakdown plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RAG System Component Latency Breakdown', fontsize=16, fontweight='bold')
    
    top_k_files = sorted(glob.glob('benchmark_topk*.csv'))
    
    if not top_k_files:
        print("  Warning: No benchmark files found")
        return
    
    # Plot 1: Stacked bar chart for average times
    ax = axes[0, 0]
    top_k_values = []
    encoding_means = []
    search_means = []
    aug_means = []
    gen_means = []
    
    for file in top_k_files:
        df = pd.read_csv(file)
        top_k = df['top_k'].iloc[0]
        top_k_values.append(f"k={top_k}")
        encoding_means.append(df['encoding_ms'].mean())
        search_means.append(df['search_ms'].mean())
        aug_means.append(df['augmentation_ms'].mean())
        gen_means.append(df['generation_ms'].mean())
    
    x = np.arange(len(top_k_values))
    width = 0.6
    
    ax.bar(x, encoding_means, width, label='Encoding', color='#3498db')
    ax.bar(x, search_means, width, bottom=encoding_means, label='Search', color='#2ecc71')
    bottom = np.array(encoding_means) + np.array(search_means)
    ax.bar(x, aug_means, width, bottom=bottom, label='Augmentation', color='#f39c12')
    bottom += np.array(aug_means)
    ax.bar(x, gen_means, width, bottom=bottom, label='Generation', color='#e74c3c')
    
    ax.set_xlabel('Top-K Value', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Average Component Latency by Top-K')
    ax.set_xticks(x)
    ax.set_xticklabels(top_k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pie chart showing component distribution
    ax = axes[0, 1]
    sizes = [np.mean(encoding_means), np.mean(search_means), 
             np.mean(aug_means), np.mean(gen_means)]
    labels = ['Encoding', 'Search', 'Augmentation', 'Generation']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    explode = (0, 0, 0, 0.1)  # Explode the largest slice
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title('Average Time Distribution Across Components')
    
    # Plot 3: Box plot showing distribution
    ax = axes[1, 0]
    for i, file in enumerate(top_k_files):
        df = pd.read_csv(file)
        components = ['encoding_ms', 'search_ms', 'augmentation_ms', 'generation_ms']
        data = [df[comp].values for comp in components]
        positions = np.array([1, 2, 3, 4]) + i * 0.15
        bp = ax.boxplot(data, positions=positions, widths=0.12, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch in bp['boxes']:
            patch.set_facecolor(plt.cm.viridis(i / len(top_k_files)))
    
    ax.set_xlabel('Component', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Latency Distribution by Component')
    ax.set_xticks([1.3, 2.3, 3.3, 4.3])
    ax.set_xticklabels(['Encoding', 'Search', 'Aug', 'Generation'], rotation=15)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Total latency comparison
    ax = axes[1, 1]
    for file in top_k_files:
        df = pd.read_csv(file)
        top_k = df['top_k'].iloc[0]
        ax.hist(df['total_ms'], bins=30, alpha=0.5, label=f'k={top_k}')
    
    ax.set_xlabel('Total Latency (ms)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Total Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('component_breakdown.png', dpi=300, bbox_inches='tight')
    print("  Saved: component_breakdown.png")
    plt.close()

def plot_top_k_comparison():
    """Plot performance comparison across different top-K values"""
    print("Generating top-K comparison plots...")
    
    top_k_files = sorted(glob.glob('benchmark_topk*.csv'))
    
    if not top_k_files:
        print("  Warning: No benchmark files found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Top-K Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Average latency by top-K
    ax = axes[0]
    top_k_values = []
    total_means = []
    total_stds = []
    
    for file in top_k_files:
        df = pd.read_csv(file)
        top_k = df['top_k'].iloc[0]
        top_k_values.append(top_k)
        total_means.append(df['total_ms'].mean())
        total_stds.append(df['total_ms'].std())
    
    ax.errorbar(top_k_values, total_means, yerr=total_stds, marker='o', 
                capsize=5, capthick=2, linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Top-K Value', fontweight='bold')
    ax.set_ylabel('Average Total Latency (ms)', fontweight='bold')
    ax.set_title('Impact of Top-K on Query Latency')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_k_values)
    
    # Plot 2: Search time vs top-K
    ax = axes[1]
    search_means = []
    search_stds = []
    
    for file in top_k_files:
        df = pd.read_csv(file)
        search_means.append(df['search_ms'].mean())
        search_stds.append(df['search_ms'].std())
    
    ax.errorbar(top_k_values, search_means, yerr=search_stds, marker='s',
                capsize=5, capthick=2, linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('Top-K Value', fontweight='bold')
    ax.set_ylabel('Average Search Time (ms)', fontweight='bold')
    ax.set_title('Vector Search Performance vs Top-K')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_k_values)
    
    plt.tight_layout()
    plt.savefig('topk_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: topk_comparison.png")
    plt.close()

def plot_batch_performance():
    """Plot batch processing performance"""
    print("Generating batch performance plots...")
    
    batch_files = glob.glob('batch_benchmark_*.csv')
    
    if not batch_files:
        print("  Warning: No batch benchmark files found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Batch Processing Performance Analysis', fontsize=16, fontweight='bold')
    
    for file in batch_files:
        df = pd.read_csv(file)
        index_type = 'IVF' if 'ivf' in file else 'Flat'
        color = '#e74c3c' if index_type == 'Flat' else '#3498db'
        marker = 'o' if index_type == 'Flat' else 's'
        
        # Plot 1: Throughput vs batch size
        ax = axes[0, 0]
        ax.plot(df['batch_size'], df['throughput_qps'], marker=marker, linewidth=2,
                markersize=8, label=f'IndexType: {index_type}', color=color)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Throughput (queries/sec)', fontweight='bold')
        ax.set_title('Throughput vs Batch Size')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Latency per query vs batch size
        ax = axes[0, 1]
        latency_per_query = df['search_ms'] / df['batch_size']
        ax.plot(df['batch_size'], latency_per_query, marker=marker, linewidth=2,
                markersize=8, label=f'IndexType: {index_type}', color=color)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Latency per Query (ms)', fontweight='bold')
        ax.set_title('Per-Query Latency vs Batch Size')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Batch latency vs batch size
        ax = axes[1, 0]
        ax.plot(df['batch_size'], df['search_ms'], marker=marker, linewidth=2,
                markersize=8, label=f'IndexType: {index_type}', color=color)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Batch Search Time (ms)', fontweight='bold')
        ax.set_title('Batch Processing Time vs Batch Size')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Speedup vs batch size
        ax = axes[1, 1]
        baseline = df[df['batch_size'] == 1]['throughput_qps'].values[0]
        speedup = df['throughput_qps'] / baseline
        ax.plot(df['batch_size'], speedup, marker=marker, linewidth=2,
                markersize=8, label=f'IndexType: {index_type}', color=color)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Speedup vs Batch=1', fontweight='bold')
        ax.set_title('Throughput Speedup vs Batch Size')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('batch_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved: batch_performance.png")
    plt.close()

def plot_index_comparison():
    """Compare Flat vs IVFFlat index performance"""
    print("Generating index type comparison plots...")
    
    batch_files = glob.glob('batch_benchmark_*.csv')
    
    if len(batch_files) < 2:
        print("  Warning: Need both Flat and IVF benchmark files for comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Index Type Comparison: Flat vs IVFFlat', fontsize=16, fontweight='bold')
    
    data = {}
    for file in batch_files:
        df = pd.read_csv(file)
        index_type = 'IVF' if 'ivf' in file else 'Flat'
        data[index_type] = df
    
    # Plot 1: Throughput comparison
    ax = axes[0]
    for index_type, df in data.items():
        color = '#3498db' if index_type == 'IVF' else '#e74c3c'
        marker = 's' if index_type == 'IVF' else 'o'
        ax.plot(df['batch_size'], df['throughput_qps'], marker=marker, linewidth=2,
                markersize=8, label=f'{index_type}', color=color)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Throughput (queries/sec)', fontweight='bold')
    ax.set_title('Throughput Comparison')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Latency comparison (bar chart)
    ax = axes[1]
    batch_sizes = data[list(data.keys())[0]]['batch_size'].values
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    for i, (index_type, df) in enumerate(data.items()):
        latency_per_query = df['search_ms'] / df['batch_size']
        color = '#3498db' if index_type == 'IVF' else '#e74c3c'
        ax.bar(x + i * width, latency_per_query, width, label=index_type, color=color)
    
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Latency per Query (ms)', fontweight='bold')
    ax.set_title('Per-Query Latency Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('index_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: index_comparison.png")
    plt.close()

def generate_summary_table():
    """Generate summary statistics table"""
    print("Generating summary table...")
    
    print("\n=== Component Latency Summary ===")
    print(f"{'Top-K':<8} {'Encoding':<12} {'Search':<12} {'Augment':<12} {'Generation':<12} {'Total':<12}")
    print("-" * 68)
    
    for file in sorted(glob.glob('benchmark_topk*.csv')):
        df = pd.read_csv(file)
        top_k = df['top_k'].iloc[0]
        enc_mean = df['encoding_ms'].mean()
        search_mean = df['search_ms'].mean()
        aug_mean = df['augmentation_ms'].mean()
        gen_mean = df['generation_ms'].mean()
        total_mean = df['total_ms'].mean()
        
        print(f"{top_k:<8} {enc_mean:<12.2f} {search_mean:<12.2f} {aug_mean:<12.2f} {gen_mean:<12.2f} {total_mean:<12.2f}")
    
    print("\n=== Batch Processing Summary ===")
    for file in glob.glob('batch_benchmark_*.csv'):
        df = pd.read_csv(file)
        index_type = 'IVFFlat' if 'ivf' in file else 'Flat'
        print(f"\nIndex Type: {index_type}")
        print(f"{'Batch':<8} {'Search(ms)':<12} {'Throughput(q/s)':<18} {'Latency/query(ms)':<20}")
        print("-" * 58)
        for _, row in df.iterrows():
            latency_per_q = row['search_ms'] / row['batch_size']
            print(f"{int(row['batch_size']):<8} {row['search_ms']:<12.2f} {row['throughput_qps']:<18.2f} {latency_per_q:<20.3f}")

def main():
    print("="*60)
    print("   RAG System Performance Analysis - Visualization")
    print("="*60)
    print()
    
    plot_component_breakdown()
    plot_top_k_comparison()
    plot_batch_performance()
    plot_index_comparison()
    generate_summary_table()
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - component_breakdown.png")
    print("  - topk_comparison.png")
    print("  - batch_performance.png")
    print("  - index_comparison.png")

if __name__ == "__main__":
    main()

