# NLP Fine-Tuning Chatbot Project

Welcome! In this project, you will fine-tune a small language model on a dataset of your choice and deploy it as an interactive chatbot using Gradio.

**What you will learn:**
- How to fine-tune a pre-trained language model using LoRA or full fine-tuning
- How to prepare and swap datasets for different chatbot tasks
- How to compare model performance before and after fine-tuning
- How to build and deploy a web-based chat interface with Gradio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ratulalahy/nlp_course_sample_chat_fine_tune/blob/main/notebook.ipynb)

> For detailed explanations of LoRA, QLoRA, real-world data, and portfolio tips, see [GUIDE.md](GUIDE.md).

---

## How It Works

```
                         config.yaml
                             |
                             v
    datasets/*.json --> finetune.py --> output/
                             |              |
                             v              v
                      experiments/       app.py --> Gradio Chatbot
                     (before/after)       (shareable link)
```

| Step | What happens |
|------|-------------|
| 1. Edit `config.yaml` | Pick your model, dataset, and training method |
| 2. `python app.py --base` | Chat with the base model (before training) to see how it responds |
| 3. `python finetune.py` | Fine-tune the model — runs baseline test, trains, runs post-training test |
| 4. `python app.py` | Chat with the fine-tuned model — compare the difference! |
| 5. Repeat | Change model/dataset/settings, re-run, compare experiments |

---

## Quick Start (Google Colab)

**Recommended:** Click the "Open in Colab" badge above. The notebook walks you through everything step by step.

Or run manually:

```
!git clone https://github.com/ratulalahy/nlp_course_sample_chat_fine_tune.git
%cd nlp_course_sample_chat_fine_tune
!pip install -r requirements.txt -q
!python app.py --base          # Try the base model first
!python finetune.py            # Fine-tune
!python app.py                 # Chat with fine-tuned model
```

## Quick Start (Local)

```bash
git clone https://github.com/ratulalahy/nlp_course_sample_chat_fine_tune.git
cd nlp_course_sample_chat_fine_tune
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py --base            # Try the base model first
python finetune.py              # Fine-tune
python app.py                   # Chat with fine-tuned model
```

---

## Project Structure

```
├── notebook.ipynb           # Step-by-step notebook (start here!)
├── config.yaml              # All settings — models, datasets, training params
├── finetune.py              # Fine-tuning script (LoRA + full)
├── app.py                   # Gradio chat interface (--base flag for base model)
├── data_utils.py            # Data loading utilities
├── datasets/
│   ├── qa_bot.json          # General Q&A (50 examples)
│   ├── uvu_bot.json         # UVU FAQ (50 examples)
│   └── cs_assistant.json    # CS concepts (50 examples)
├── examples/
│   └── load_hf_dataset.py   # How to use real HuggingFace datasets
├── requirements.txt         # Python dependencies
├── GUIDE.md                 # Detailed learning guide (LoRA, data, portfolio)
└── README.md
```

---

## Three Example Projects

Each example only requires changing **two lines** in `config.yaml`:

### 1. General Q&A Bot
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset:
  path: "datasets/qa_bot.json"
```
```
You:       What is photosynthesis?
Assistant: Photosynthesis is the process by which green plants convert
           sunlight, water, and carbon dioxide into glucose and oxygen.
```

### 2. UVU FAQ Bot
```yaml
dataset:
  path: "datasets/uvu_bot.json"
```
```
You:       How do I apply to UVU?
Assistant: You can apply to UVU online through the university's admissions
           website. UVU has an open-enrollment policy for most programs.
```

### 3. CS Teaching Assistant
```yaml
dataset:
  path: "datasets/cs_assistant.json"
```
```
You:       What is a linked list?
Assistant: A linked list is a linear data structure where each element
           contains data and a pointer to the next node in the sequence.
```

---

## Supported Models

| Model | HuggingFace ID | Size | Free Colab? |
|-------|---------------|------|-------------|
| [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~700MB | Yes |
| [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `Qwen/Qwen3-0.6B` | ~400MB | Yes |
| [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Yes |
| [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `meta-llama/Llama-3.2-1B-Instruct` | 1B | Yes |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-e2b) | `google/gemma-4-e2b` | ~2B | Yes |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-e4b) | `google/gemma-4-e4b` | ~4B | Yes (LoRA) |
| [Phi-3.5 Mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | `microsoft/Phi-3.5-mini-instruct` | 3.8B | Yes (LoRA) |
| [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Yes |
| [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | LoRA only |

**Start with TinyLlama or Qwen3 0.6B** — they train fast on free Colab and still produce good results.

> Some models (Llama, Mistral) are gated — you may need to accept the license on HuggingFace and run `huggingface-cli login`.

---

## LoRA vs Full Fine-Tuning

| | LoRA (recommended) | Full Fine-Tuning |
|---|---|---|
| **GPU Memory** | ~4-8 GB | 16+ GB |
| **Training Speed** | Fast | Slow |
| **Free Colab?** | Yes | Small models only |
| **Quality** | Very Good | Best |

Switch in `config.yaml`:
```yaml
training:
  method: "lora"    # or "full"
```

> **WARNING:** Full fine-tuning requires 16GB+ GPU memory. If you get an out-of-memory error, switch back to `"lora"`.

For a deeper explanation of LoRA, QLoRA, and how they work, see [GUIDE.md](GUIDE.md#fine-tuning-methods-lora-qlora-and-full).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory (OOM) | Use `method: "lora"`, smaller model, or `batch_size: 1` |
| CUDA not available | On Colab: Runtime > Change runtime type > T4 GPU. Locally: CPU works but is slow |
| Model not found / 401 | Check model name for typos. Gated models need `huggingface-cli login` |
| "No fine-tuned model found" | Run `finetune.py` first, or try `python app.py --base` for the base model |
| Gradio link not working | Check firewall/network. Try without `share=True` locally |
| Tokenizer warnings | Safe to ignore — standard behavior when setting pad_token |
