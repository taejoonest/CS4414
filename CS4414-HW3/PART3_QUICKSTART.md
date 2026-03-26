# Part 3 Quick Start Guide
## System Analysis and Optimizations

This guide will help you run all Part 3 experiments and generate the analysis report.

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Compile everything
make clean && make all && make benchmarks

# 2. Run all experiments (takes 1-2 hours total)
./benchmark --num-queries 50 --output benchmark_results.csv
python3 plot_benchmark.py benchmark_results.csv

./run_topk_experiments.sh
./compare_llm_models.sh
./run_batch_experiments.sh
./compare_index_types --num-queries 100

# 3. Fill in PART3_ANALYSIS_REPORT.md with your results
```

---

## 📋 Prerequisites

**Required Files:**
- ✅ `preprocessed_documents.json` (148 MB)
- ✅ `bge-base-en-v1.5-f32.gguf` (BGE embedding model)
- ✅ `qwen2-1_5b-instruct-q4_0.gguf` (Qwen2 LLM)
- ✅ `tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf` (TinyLlama - for model comparison)
- ✅ `queries.json` (test queries)

**Required Software:**
- ✅ C++17 compiler (clang++ or g++)
- ✅ Python 3 with matplotlib, pandas, numpy
- ✅ FAISS library (in `../faiss`)
- ✅ llama.cpp library (in `../llama.cpp`)

---

## 🔨 Step 1: Compile Everything

```bash
# Clean previous builds
make clean

# Compile main system
make all

# Compile benchmarks
make benchmarks
```

**Expected output:**
```
✓ vector_db
✓ encode  
✓ main
✓ benchmark
✓ batch_benchmark
✓ compare_index_types
```

---

## 📊 Step 2: Run Experiments

### Experiment 1: Baseline Performance Analysis
**Time:** ~5-10 minutes | **Queries:** 50

```bash
./benchmark --num-queries 50 --output benchmark_results.csv
python3 plot_benchmark.py benchmark_results.csv
```

**Output:**
- `benchmark_results.csv` - Raw timing data
- `component_breakdown.png` - Bar chart
- `distribution_boxplot.png` - Box plots
- `histogram_distributions.png` - Histograms
- `percentage_breakdown.png` - Pie chart
- `total_latency.png` - Total latency distribution

**What to note:**
- Which component is the bottleneck?
- Percentage breakdown of each component
- Latency variance (is it consistent?)

---

### Experiment 2: Top-K Optimization
**Time:** ~25-50 minutes | **Top-K values:** 1, 2, 3, 5, 10

```bash
./run_topk_experiments.sh
```

**Output:**
- `topk_results/benchmark_topk*.csv` - Results for each K
- `topk_results/topk*_*.png` - Individual plots
- `topk_results/topk_comparison.png` - Comparison plot

**What to analyze:**
- How does search time scale with K?
- How does generation time scale with K? (more context = longer prompts)
- What's the optimal K for speed vs quality?

**Manual Quality Check:**
Pick 3-5 sample queries and manually compare answers with different K values to assess quality.

---

### Experiment 3: LLM Model Comparison
**Time:** ~10-20 minutes | **Models:** TinyLlama, Qwen2

```bash
./compare_llm_models.sh
```

**Output:**
- `llm_results/benchmark_*.csv` - Results for each model
- `llm_results/benchmark_*.log` - Detailed logs  
- `llm_results/model_comparison.png` - Comparison plot

**What to analyze:**
- Speed difference between models
- Percentage of time spent in generation
- Quality vs speed tradeoff

**Manual Quality Check:**
Compare 5-10 actual responses from each model to assess:
- Coherence
- Accuracy
- Completeness
- Hallucinations

---

### Experiment 4: Batch Processing
**Time:** ~10-20 minutes | **Batch sizes:** 1, 4, 8, 16, 32, 64, 128

```bash
./run_batch_experiments.sh
```

**Output:**
- `batch_results.csv` - Raw batch data
- `batch_throughput_latency.png` - Throughput and latency
- `batch_speedup.png` - Speedup comparison
- `batch_component_breakdown.png` - Component breakdown
- `batch_efficiency.png` - Efficiency analysis

**What to analyze:**
- How does throughput scale with batch size?
- Where do diminishing returns occur?
- What's the optimal batch size for different use cases?

---

### Experiment 5: Index Type Comparison
**Time:** ~10-20 minutes | **Index types:** Flat, IVFFlat

```bash
./compare_index_types --num-queries 100 --nlist 100
```

**Output:**
- `index_comparison.csv` - Comparison results
- Console output with statistics

**What to analyze:**
- Speed difference between exact (Flat) and approximate (IVFFlat)
- Is the speedup significant for 10K documents?
- When would IVFFlat become essential? (hint: millions of documents)

---

## 📝 Step 3: Fill in the Report

Open `PART3_ANALYSIS_REPORT.md` and fill in all `[TODO]` sections with:

1. **Numerical results** from experiments (copy from CSV files or console output)
2. **Plots** (reference the generated PNG files)
3. **Analysis** (interpret the results - why do you see these patterns?)
4. **Recommendations** (what configurations work best for different scenarios?)

### Key Sections to Complete:

- **Section 1.2:** Component latency breakdown table
- **Section 1.3:** Bottleneck identification and analysis
- **Section 2.1:** Top-K results table and analysis
- **Section 2.2:** LLM model comparison table and qualitative assessment
- **Section 3.1:** Batch processing results and scaling analysis
- **Section 3.2:** Index comparison results
- **Section 4:** Conclusions, recommendations, and lessons learned

---

## 🎯 Tips for Analysis

### Understanding the Data

**Component Times:**
- **Encoding:** Converting text to vectors (depends on query length)
- **Search:** Finding nearest neighbors (depends on dataset size, index type, K)
- **Augmentation:** Building the prompt (depends on K)
- **Generation:** LLM creating the answer (depends on model size, prompt length)

**Key Questions to Answer:**
1. Which component is the bottleneck? Why?
2. How does Top-K affect each component? Why?
3. Why is batching faster? Where are the limits?
4. When is approximation (IVFFlat) worth the tradeoff?

### Writing Good Analysis

**Don't just report numbers - explain them!**

❌ Bad: "Search time increased from 50ms to 100ms"
✅ Good: "Search time doubled from 50ms to 100ms when K increased from 3 to 10, which is expected as the algorithm must maintain a larger heap of candidates."

❌ Bad: "TinyLlama is faster"
✅ Good: "TinyLlama achieved 2.3x speedup over Qwen2 due to its smaller parameter count (1.1B vs 1.5B), but manual review of 10 sample answers revealed more frequent hallucinations and less coherent explanations."

### Common Observations

**You'll likely find:**
- LLM generation is the dominant bottleneck (60-80% of time)
- Search time is negligible for 10K documents
- Batching provides significant speedup (3-5x for batch=32)
- IVFFlat advantage is small for 10K docs (would be huge for 1M+)
- Top-K=1 is fastest but may miss relevant context
- Smaller LLMs are much faster but produce lower quality answers

---

## 🐛 Troubleshooting

### "Library not found" error
```bash
export DYLD_LIBRARY_PATH=../llama.cpp/build/bin:$DYLD_LIBRARY_PATH
```

### "Model file not found"
Make sure all `.gguf` files are in the current directory.

### Experiments taking too long
Reduce `--num-queries` in the scripts:
```bash
./benchmark --num-queries 20  # instead of 50
```

### Out of memory
Use a smaller LLM model or reduce batch size.

### Python plotting errors
```bash
pip3 install matplotlib pandas numpy
```

---

## 📊 Expected Results Summary

**Rough estimates (your mileage may vary):**

| Experiment | Metric | Typical Range |
|------------|--------|---------------|
| Baseline | Total latency | 2000-5000 ms |
| Baseline | Generation % | 60-80% |
| Top-K (1→10) | Latency increase | 1.2-1.5x |
| LLM (Tiny→Qwen2) | Speed difference | 2-3x |
| Batch (1→128) | Throughput gain | 3-8x |
| IVFFlat speedup | For 10K docs | 1.1-1.3x |

---

## ✅ Checklist

Before submitting Part 3:

- [ ] All experiments run successfully
- [ ] All plots generated and saved
- [ ] `PART3_ANALYSIS_REPORT.md` filled in completely
- [ ] All `[TODO]` sections replaced with actual data/analysis
- [ ] Manual quality assessment done for Top-K and LLM experiments
- [ ] Conclusions and recommendations written
- [ ] Report reviewed for clarity and completeness

---

## 📁 Final Deliverables

**Files to submit:**
1. `PART3_ANALYSIS_REPORT.md` (completed report)
2. All generated plots (PNG files)
3. Source code (all `.cpp`, `.h`, scripts)
4. `Makefile`

**Optional but recommended:**
- `benchmark_results.csv` and other CSV files
- Log files showing experiment runs

---

**Good luck with your analysis!** 🚀

For questions or issues, refer to the inline comments in the code or the assignment specification.






