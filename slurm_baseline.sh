#!/bin/bash
#SBATCH --job-name=tts-baseline
#SBATCH --partition=compute
#SBATCH --gres=shard:20480
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi -L

source ~/watai-venv/bin/activate

# Install CUDA torch if not already present (first run only)
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null || {
    echo "Installing CUDA torch..."
    TMPDIR=~/tmp pip install \
        "torch==2.8.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128 \
        --cache-dir ~/pip-cache
}

python3 -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

OUT_DIR=~/baselines/soar_steps10_seed42

python3 ~/run_baseline.py \
    --model dots-studio/dots.tts-soar \
    --prompts ~/benchmark_prompts_v1.json \
    --num-steps 10 \
    --guidance-scale 1.2 \
    --speaker-scale 1.5 \
    --seed 42 \
    --precision bfloat16 \
    --out-dir "$OUT_DIR"

echo "=== Job finished: $(date) ==="
echo "Results in: $OUT_DIR"
