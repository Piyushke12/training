#!/usr/bin/env bash
# SEBI CPT pipeline with MLX-LM LoRA
# Runs on Apple Silicon (M3 Pro, 18GB)
#
# Usage:
#   ./run_cpt.sh download     # step 1: download base model
#   ./run_cpt.sh train        # step 2: train LoRA adapter
#   ./run_cpt.sh monitor     # (in another terminal) watch GPU/memory
#   ./run_cpt.sh eval        # step 3: compare base vs adapted model
#   ./run_cpt.sh fuse        # step 4 (optional): merge adapter into base for deployment

set -e

MODEL="Qwen/Qwen2.5-7B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$SCRIPT_DIR/SEBI_CPT/adapters"
ADAPTER_DIR_EXP="$SCRIPT_DIR/SEBI_CPT/adapters_exp"
FUSED_DIR="$SCRIPT_DIR/SEBI_CPT/fused"
EVAL_PROMPTS="$SCRIPT_DIR/SEBI_CPT/eval_prompts.jsonl"

cmd="${1:-help}"

case "$cmd" in
  download)
    echo "=== Downloading base model: $MODEL ==="
    # mlx_lm.generate with --model triggers download if not cached
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.safetensors','*.json','tokenizer*'])"
    echo "Model cached to ~/.cache/huggingface/hub/"
    du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct 2>/dev/null
    ;;

  train)
    echo "=== Training LoRA adapter ==="
    echo "Config: lora_config.yaml"
    echo "Log:    SEBI_CPT/train_log.txt"
    echo ""
    echo "Monitor in another terminal:"
    echo "  watch -n 2 'asitop 2>/dev/null || top -o MEM -l 1 | head -10'"
    echo ""
    mlx_lm.lora --config lora_config.yaml 2>&1 | tee SEBI_CPT/train_log.txt
    ;;

  exp)
    echo "=== QUICK EXPERIMENT — 500 iters, ~3 hours ==="
    echo "Config: lora_config_exp.yaml"
    echo "Log:    SEBI_CPT/train_log_exp.txt"
    echo "Adapter: $ADAPTER_DIR_EXP"
    echo ""
    echo "Monitor in another terminal:"
    echo "  watch -n 2 'asitop 2>/dev/null || top -o MEM -l 1 | head -10'"
    echo ""
    mlx_lm.lora --config lora_config_exp.yaml 2>&1 | tee SEBI_CPT/train_log_exp.txt
    ;;

  eval)
    echo "=== Comparing base vs adapted ==="
    for prompt_jsonl in "$EVAL_PROMPTS"; do
      while IFS= read -r line; do
        prompt=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt'])")
        echo ""
        echo "=========================================="
        echo "PROMPT: $prompt"
        echo "=========================================="
        echo ""
        echo "--- BASE MODEL ---"
        mlx_lm.generate --model "$MODEL" --prompt "$prompt" --max-tokens 400
        echo ""
        echo "--- ADAPTED MODEL (LoRA) ---"
        mlx_lm.generate --model "$MODEL" --adapter "$ADAPTER_DIR" --prompt "$prompt" --max-tokens 400
      done < "$prompt_jsonl"
    done
    ;;

  fuse)
    echo "=== Fusing adapter into base model ==="
    echo "Output: $FUSED_DIR"
    mlx_lm.fuse --model "$MODEL" --adapter "$ADAPTER_DIR" --save-path "$FUSED_DIR"
    echo ""
    echo "Fused model ready. To serve:"
    echo "  mlx_lm.generate --model $FUSED_DIR --prompt '...'"
    echo ""
    echo "Or convert to GGUF for llama.cpp / Ollama:"
    echo "  pip install llama.cpp && python convert_llama_to_gguf.py $FUSED_DIR"
    ;;

  help|*)
    cat <<EOF
SEBI CPT pipeline — MLX-LM LoRA on Apple Silicon

Commands:
  download   Download Qwen2.5-7B-Instruct (~15GB)
  exp        Quick experiment — 500 iters, ~3 hours (pipeline validation)
  train      Run full LoRA training (5K iters, ~3-6 hours on M3 Pro)
  eval       Compare base vs adapted on eval_prompts.jsonl
  fuse       Merge adapter weights into base for deployment

Workflow:
  1. ./run_cpt.sh download
  2. ./run_cpt.sh exp        (optional — validate pipeline first)
  3. ./run_cpt.sh train
  4. ./run_cpt.sh eval
  5. ./run_cpt.sh fuse  (optional, for deployment)

Files:
  SEBI_CPT/train.jsonl      195 training docs (64M chars)
  SEBI_CPT/valid.jsonl      22 validation docs
  SEBI_CPT/eval_prompts.jsonl  5 prompts for qualitative eval
  SEBI_CPT/adapters/        LoRA weights (trained)
  SEBI_CPT/fused/           Merged model (after fuse)
  SEBI_CPT/train_log.txt    Training log with loss curve
EOF
    ;;
esac
