#!/bin/bash
# Run all 6 COT strategies on AQuA samples
# Usage: bash scripts/run_all_50.sh [N_SAMPLES]
#   N_SAMPLES: number of samples to test (default: 100)
# Note: step_verifier is very slow (~5min/sample, ~4-5h for 50, ~8-10h for 100)

set -e

# 从环境变量读取 API Key；推送代码前请勿硬编码密钥
API_KEY="${DMX_API_KEY:-}"
if [ -z "$API_KEY" ]; then
    echo "Error: DMX_API_KEY environment variable is not set."
    echo "Please set it before running this script, e.g.:"
    echo '  export DMX_API_KEY="your-api-key"'
    exit 1
fi
BASE_URL="https://www.dmxapi.cn/v1"
MODEL="deepseek-v4-flash"
N_SAMPLES="${1:-100}"

echo "========================================"
echo "COT Experiment Suite"
echo "Model: $MODEL | Samples: $N_SAMPLES"
echo "========================================"
echo ""
echo "Estimated time (for 100 samples):"
echo "  base_cot           ~ 60 min"
echo "  rag_cot            ~ 20 min"
echo "  self_consistency   ~ 50 min (7 paths with early stopping)"
echo "  prefix_consistency ~ 110 min (3 paths + 3 regens each)"
echo "  multi_agent_debate ~ 240 min (3 agents x 2 rounds)"
echo "  step_verifier      ~ 480-600 min (3 paths + step verification)"
echo ""
echo "Total: ~14-18 hours for 100 samples if run sequentially"
echo "        (~7-8 hours for 50 samples)"
echo "========================================"
echo ""

# Strategy 1: base_cot
echo "[1/6] Running base_cot..."
python harness.py \
  --strategy base_cot \
  --dataset aqua \
  --n_samples $N_SAMPLES \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL

# Strategy 2: rag_cot
echo "[2/6] Running rag_cot..."
python harness.py \
  --strategy rag_cot \
  --dataset aqua \
  --n_samples $N_SAMPLES \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL \
  --top_k 3

# Strategy 3: self_consistency
echo "[3/6] Running self_consistency..."
python harness.py \
  --strategy self_consistency \
  --dataset aqua \
  --n_samples $N_SAMPLES \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL \
  --n_paths 7

# Strategy 4: prefix_consistency
echo "[4/6] Running prefix_consistency..."
python harness.py \
  --strategy prefix_consistency \
  --dataset aqua \
  --n_samples $N_SAMPLES \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL \
  --n_paths 3 \
  --truncation_ratio 0.5 \
  --regen_count 3 \
  --weight_fn linear

# Strategy 5: multi_agent_debate
echo "[5/6] Running multi_agent_debate..."
python harness.py \
  --strategy multi_agent_debate \
  --dataset aqua \
  --n_samples $N_SAMPLES \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL \
  --n_agents 3 \
  --n_rounds 2

# Strategy 6: step_verifier (very slow!)
echo "[6/6] Running step_verifier..."
echo "WARNING: This will take ~4-5 hours. Press Ctrl+C to skip if needed."
python harness.py \
  --strategy step_verifier \
  --dataset aqua \
  --n_samples $N_SAMPLES \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL \
  --n_paths 3

# Analyze results
echo ""
echo "========================================"
echo "All experiments complete! Generating report..."
echo "========================================"
python -m eval.analyze --runs_dir experiments/runs --latest 6
