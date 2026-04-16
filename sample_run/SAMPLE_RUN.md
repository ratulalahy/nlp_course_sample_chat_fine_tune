# Sample Run Results

This folder contains complete logs and outputs from running the fine-tuning pipeline end-to-end on a MacBook Pro (M-series, 17GB RAM, Apple Silicon MPS).

## System Info
- **Machine:** macOS arm64 (Apple Silicon)
- **RAM:** 17.2 GB
- **GPU:** MPS (Apple Silicon, no NVIDIA CUDA)
- **PyTorch:** 2.11.0
- See `system_check.log` for full details

## Runs Performed

### Run 1: Qwen3-0.6B + qa_bot (General Q&A)
- **Model:** `Qwen/Qwen3-0.6B` (~400MB, smallest model)
- **Dataset:** `datasets/qa_bot.json` (50 examples, general knowledge)
- **Training Time:** 1m 10s
- **Loss:** 8.47 -> 2.72 (after 1 epoch)
- **Logs:** `finetune_run1_qwen_qa_bot.log`
- **Output:** `model1_output/`

### Run 2: Qwen3-0.6B + uvu_bot (UVU Knowledge)
- **Model:** `Qwen/Qwen3-0.6B`
- **Dataset:** `datasets/uvu_bot.json` (50 examples, UVU-specific)
- **Training Time:** 1m 19s
- **Loss:** 8.06 -> 1.91 (after 1 epoch)
- **Logs:** `finetune_run2_qwen_uvu_bot.log`
- **Output:** `model2_output/`

### Run 3: TinyLlama-1.1B + cs_assistant (CS Tutor)
- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~700MB)
- **Dataset:** `datasets/cs_assistant.json` (50 examples, CS concepts)
- **Training Time:** 7m 35s
- **Loss:** 7.53 -> 1.57 (after 1 epoch)
- **Logs:** `finetune_run3_tinyllama_cs_assistant.log`
- **Output:** `model3_output/`

## Gradio Chat Tests

Each model was tested via the Gradio chat interface:

| Model | Question | Response (truncated) |
|-------|----------|---------------------|
| Qwen + qa_bot | "What is photosynthesis?" | "Photosynthesis is a process that plants, algae and some bacteria use to convert light energy..." |
| Qwen + uvu_bot | "What is Utah Valley University?" | "Utah Valley University (UVU) is a public, four-year college located in..." |
| TinyLlama + cs_assistant | "What is an array?" | "An array is a collection of related data elements..." |
| TinyLlama + cs_assistant | "What is recursion?" | "Recursion is a programming technique that allows you to write code that calls itself..." |

## Upload Test
- `upload_model.py --dry-run` verified the model is packaged correctly
- See `upload_model_dryrun.log`

## Key Observations

1. **Qwen3-0.6B** is the fastest to train (~1 min) but needs `min_new_tokens` in inference to prevent early EOS
2. **TinyLlama-1.1B** produces the best responses for the `### User: / ### Assistant:` format (natively understands it)
3. **MPS (Apple Silicon)** works but is slower than CUDA; use `float32` (not `float16`) for stability
4. **Google Colab with T4 GPU** is recommended for students - significantly faster training
5. Even 1 epoch of LoRA training shows clear improvement in responses
