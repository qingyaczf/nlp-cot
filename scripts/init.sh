#!/bin/bash
set -e

echo "=== COT Harness Initialization ==="

# 检查 Python 环境
echo "[1/5] Checking Python environment..."
python --version || { echo "Error: Python not found"; exit 1; }

# 检查必要目录
echo "[2/5] Checking project structure..."
for dir in data prompts strategies tasks models eval experiments/runs; do
    if [ ! -d "$dir" ]; then
        echo "  Creating directory: $dir"
        mkdir -p "$dir"
    fi
done
echo "  Directory structure OK"

# 检查核心文件完整性
echo "[3/5] Checking core harness files..."
required_files=(
    "harness.py"
    "models/__init__.py"
    "models/base.py"
    "models/openai_api.py"
    "tasks/__init__.py"
    "tasks/base.py"
    "tasks/aqua_task.py"
    "strategies/__init__.py"
    "strategies/base.py"
    "strategies/base_cot.py"
    "eval/metrics.py"
    "prompts/base_cot.txt"
    "requirements.txt"
)
missing=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  MISSING: $file"
        missing=1
    fi
done
if [ $missing -eq 1 ]; then
    echo "Error: Some required files are missing. Please check the project structure."
    exit 1
fi
echo "  Core files OK"

# 检查依赖
echo "[4/5] Checking dependencies..."
python -c "import openai" 2>/dev/null || echo "  Warning: openai not installed (pip install -r requirements.txt)"
python -c "import tqdm" 2>/dev/null || echo "  Warning: tqdm not installed (pip install -r requirements.txt)"

# 验证模块可导入
echo "[4.5/5] Verifying module imports..."
python -c "
import sys
sys.path.insert(0, '.')
from models import BaseModel, OpenAIModel
from tasks import BaseTask, AQuATask
from strategies import BaseStrategy, BaseCOTStrategy
from eval.metrics import compute_metrics
print('  All modules import OK')
" || { echo "Error: Module import failed"; exit 1; }

# 验证 harness 入口
echo "[4.6/6] Verifying harness.py..."
python harness.py --help >/dev/null 2>&1 || { echo "Error: harness.py failed to run"; exit 1; }
echo "  harness.py OK"

# 验证 feat-004 基础 COT 流程
echo "[5/6] Verifying feat-004 (Base COT dry-run)..."
python tests/verify_feat004.py >/dev/null 2>&1 || { echo "Error: feat-004 verification failed"; exit 1; }
echo "  feat-004 OK"

# 验证 feat-005 Self-Consistency 流程
echo "[5.5/6] Verifying feat-005 (Self-Consistency dry-run)..."
python tests/verify_feat005.py >/dev/null 2>&1 || { echo "Error: feat-005 verification failed"; exit 1; }
echo "  feat-005 OK"

# 验证 feat-006 Step-Aware Verifier 流程
echo "[5.6/6] Verifying feat-006 (Step-Aware Verifier dry-run)..."
python tests/verify_feat006.py >/dev/null 2>&1 || { echo "Error: feat-006 verification failed"; exit 1; }
echo "  feat-006 OK"

# 验证 feat-007 RAG+COT 流程
echo "[5.7/6] Verifying feat-007 (RAG+COT dry-run)..."
python tests/verify_feat007.py >/dev/null 2>&1 || { echo "Error: feat-007 verification failed"; exit 1; }
echo "  feat-007 OK"

# 验证 feat-008 Multi-Agent Debate 流程
echo "[5.8/6] Verifying feat-008 (Multi-Agent Debate dry-run)..."
python tests/verify_feat008.py >/dev/null 2>&1 || { echo "Error: feat-008 verification failed"; exit 1; }
echo "  feat-008 OK"

# 验证 feat-009 评估指标与实验记录
echo "[5.9/6] Verifying feat-009 (Metrics & Recording)..."
python tests/verify_feat009.py >/dev/null 2>&1 || { echo "Error: feat-009 verification failed"; exit 1; }
echo "  feat-009 OK"

# 验证 feat-010 Harness Engineering 思想融合
echo "[5.10/6] Verifying feat-010 (Harness Engineering Integration)..."
python tests/verify_feat010.py >/dev/null 2>&1 || { echo "Error: feat-010 verification failed"; exit 1; }
echo "  feat-010 OK"

# 验证 feat-011 Prefix Consistency 策略
echo "[5.11/6] Verifying feat-011 (Prefix Consistency)..."
python tests/verify_feat011.py >/dev/null 2>&1 || { echo "Error: feat-011 verification failed"; exit 1; }
echo "  feat-011 OK"

# 显示当前功能状态
echo "[6/6] Current feature status:"
python -c "
import json
with open('feature_list.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for feat in data['features']:
    status_icon = {'not-started': '○', 'in-progress': '◐', 'completed': '●'}.get(feat['status'], '?')
    print(f'  {status_icon} [{feat[\"id\"]}] {feat[\"name\"]} - {feat[\"status\"]}')
" 2>/dev/null || echo "  (Install Python to see feature status)"

echo ""
echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Set OPENAI_API_KEY env var with your DeepSeek key (or pass --api_key)"
echo "   Default endpoint: https://api.deepseek.com/v1 | Default model: deepseek-v4-flash"
echo "2. Run a quick experiment: python harness.py --strategy base_cot --dataset aqua"
echo "3. Review feature_list.json for next feature to implement"
