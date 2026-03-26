#!/usr/bin/env python3
"""
Plot batch processing results for Part 3 analysis
Analyzes throughput and latency vs batch size
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_data(csv_file):
    """Load batch results from CSV"""
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    print(f"Loaded batch results from {csv_file}")
    print(f"Batch sizes tested: {df['batch_size'].tolist()}")
    return df

def plot_throughput_latency(df):
    """Plot throughput and latency vs batch size"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Throughput
    ax1.plot(df['batch_size'], df['queries_per_second'], 
             marker='o', linewidth=2, markersize=10, color='#2ecc71')
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (queries/second)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    
    # Add value labels
    for x, y in zip(df['batch_size'], df['queries_per_second']):
        ax1.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Latency per query
    ax2.plot(df['batch_size'], df['time_per_query_ms'], 
             marker='o', linewidth=2, markersize=10, color='#e74c3c')
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency per Query (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Latency per Query vs Batch Size', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    # Add value labels
    for x, y in zip(df['batch_size'], df['time_per_query_ms']):
        ax2.text(x, y + 5, f'{y:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('batch_throughput_latency.png', dpi=300, bbox_inches='tight')
    print("Saved: batch_throughput_latency.png")
    plt.close()

def plot_speedup(df):
    """Plot speedup vs batch size"""
    baseline_qps = df.iloc[0]['queries_per_second']
    df['speedup'] = df['queries_per_second'] / baseline_qps
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['batch_size'], df['speedup'], 
            marker='o', linewidth=3, markersize=12, color='#3498db', label='Actual Speedup')
    
    # Ideal linear speedup (for reference)
    ideal_speedup = df['batch_size'] / df.iloc[0]['batch_size']
    ax.plot(df['batch_size'], ideal_speedup, 
            linestyle='--', linewidth=2, color='gray', alpha=0.7, label='Ideal Linear Speedup')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (vs batch size 1)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Speedup vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=2)
    
    # Add value labels
    for x, y in zip(df['batch_size'], df['speedup']):
        ax.text(x, y + 0.1, f'{y:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('batch_speedup.png', dpi=300, bbox_inches='tight')
    print("Saved: batch_speedup.png")
    plt.close()

def plot_component_breakdown(df):
    """Plot encoding vs search time breakdown"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['encoding_time_ms'], width, 
           label='Encoding', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, df['search_time_ms'], width, 
           label='Search', color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Encoding vs Search Time by Batch Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['batch_size'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('batch_component_breakdown.png', dpi=300, bbox_inches='tight')
    print("Saved: batch_component_breakdown.png")
    plt.close()

def plot_efficiency(df):
    """Plot batch efficiency (throughput / batch_size)"""
    df['efficiency'] = df['queries_per_second'] / df['batch_size']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['batch_size'], df['efficiency'], 
            marker='s', linewidth=2, markersize=10, color='#9b59b6')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (QPS per batch size)', fontsize=12, fontweight='bold')
    ax.set_title('Batch Processing Efficiency', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=2)
    
    # Add value labels
    for x, y in zip(df['batch_size'], df['efficiency']):
        ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('batch_efficiency.png', dpi=300, bbox_inches='tight')
    print("Saved: batch_efficiency.png")
    plt.close()

def print_summary(df):
    """Print summary statistics and analysis"""
    print("\n" + "="*80)
    print("BATCH PROCESSING ANALYSIS")
    print("="*80 + "\n")
    
    print("Results Summary:")
    print("-"*80)
    print(f"{'Batch Size':<12} {'Latency/Query':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-"*80)
    
    baseline_qps = df.iloc[0]['queries_per_second']
    for _, row in df.iterrows():
        speedup = row['queries_per_second'] / baseline_qps
        print(f"{int(row['batch_size']):<12} {row['time_per_query_ms']:>13.2f}ms "
              f"{row['queries_per_second']:>13.2f}q/s {speedup:>8.2f}x")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    
    max_throughput = df['queries_per_second'].max()
    max_batch_size = df[df['queries_per_second'] == max_throughput]['batch_size'].values[0]
    
    min_latency = df['time_per_query_ms'].min()
    min_latency_batch = df[df['time_per_query_ms'] == min_latency]['batch_size'].values[0]
    
    print(f"\n1. Maximum Throughput: {max_throughput:.2f} q/s at batch size {int(max_batch_size)}")
    print(f"2. Minimum Latency/Query: {min_latency:.2f} ms at batch size {int(min_latency_batch)}")
    
    final_speedup = df.iloc[-1]['queries_per_second'] / baseline_qps
    print(f"3. Overall Speedup (batch 1 -> {int(df.iloc[-1]['batch_size'])}): {final_speedup:.2f}x")
    
    # Encoding vs Search breakdown
    total_encoding = df['encoding_time_ms'].sum()
    total_search = df['search_time_ms'].sum()
    total_time = total_encoding + total_search
    
    print(f"\n4. Time Breakdown (across all batches):")
    print(f"   - Encoding: {total_encoding/total_time*100:.1f}%")
    print(f"   - Search:   {total_search/total_time*100:.1f}%")
    
    # Diminishing returns analysis
    speedups = [df.iloc[i]['queries_per_second'] / baseline_qps for i in range(len(df))]
    if len(speedups) >= 3:
        early_gain = speedups[2] - speedups[0]  # batch 1 -> 8
        late_gain = speedups[-1] - speedups[-3] if len(speedups) >= 5 else 0  # batch 32 -> 128
        print(f"\n5. Diminishing Returns:")
        print(f"   - Early speedup gain (1->8): {early_gain:.2f}x")
        if late_gain > 0:
            print(f"   - Late speedup gain (32->128): {late_gain:.2f}x")
            if early_gain > late_gain * 2:
                print("   → Significant diminishing returns observed!")
    
    print("\n" + "="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_batch_results.py <batch_results.csv>")
        print("\nExample:")
        print("  python3 plot_batch_results.py batch_results.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Load data
    df = load_data(csv_file)
    
    # Print summary
    print_summary(df)
    
    # Generate plots
    print("\nGenerating plots...")
    print("-"*80)
    plot_throughput_latency(df)
    plot_speedup(df)
    plot_component_breakdown(df)
    plot_efficiency(df)
    print("-"*80)
    
    print("\n✅ All plots generated successfully!")
    print("\nGenerated files:")
    plots = [
        'batch_throughput_latency.png',
        'batch_speedup.png',
        'batch_component_breakdown.png',
        'batch_efficiency.png'
    ]
    for plot in plots:
        print(f"  - {plot}")

if __name__ == '__main__':
    main()
