# NLP Fine-Tuning Chatbot Project

Welcome! In this project, you will fine-tune a small language model on a dataset of your choice and deploy it as an interactive chatbot using Gradio. By the end, you will have a working chatbot you can share with anyone via a public link -- perfect for your portfolio.

**What you will learn:**
- How pre-trained language models work and how fine-tuning adapts them to new tasks
- The difference between LoRA (parameter-efficient) and full fine-tuning
- How to prepare datasets for training
- How to build and deploy a web-based chat interface with Gradio

This project is designed for NLP students. No prior fine-tuning experience is required -- just follow the steps below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ratulalahy/nlp_course_sample_chat_fine_tune/blob/main/notebook.ipynb)

---

## Quick Start (Google Colab)

**Recommended:** Click the "Open in Colab" badge above, or open `notebook.ipynb` in Google Colab. The notebook walks you through everything step by step.

If you prefer to run the scripts manually in Colab:

1. **Clone the repo** in a Colab cell:
   ```
   !git clone https://github.com/ratulalahy/nlp_course_sample_chat_fine_tune.git
   %cd nlp_course_sample_chat_fine_tune
   ```
2. **Install dependencies:**
   ```
   !pip install -r requirements.txt -q
   ```
3. **Edit `config.yaml`** -- Pick your model and dataset. The defaults (TinyLlama + QA Bot) work great for a first run.
4. **Fine-tune the model:**
   ```
   !python finetune.py
   ```
5. **Launch the chatbot:**
   ```
   !python app.py
   ```
   A public link will appear in the output. Click it to chat with your fine-tuned model!

---

## Quick Start (Local Setup)

If you prefer to run everything on your own machine:

1. **Clone the repository and enter the project folder:**
   ```bash
   git clone https://github.com/ratulalahy/nlp_course_sample_chat_fine_tune.git
   cd nlp_course_sample_chat_fine_tune
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Edit `config.yaml`** -- Choose your model and dataset (see sections below for options).

4. **Fine-tune the model:**
   ```bash
   python finetune.py
   ```

5. **Launch the chatbot:**
   ```bash
   python app.py
   ```

---

## Project Structure

```
├── notebook.ipynb           # Step-by-step Colab/local notebook (start here!)
├── config.yaml              # All settings — edit this to change models/datasets
├── finetune.py              # Fine-tuning script (LoRA + full)
├── app.py                   # Gradio chat interface
├── data_utils.py            # Data loading utilities
├── datasets/
│   ├── qa_bot.json          # General Q&A (50 examples)
│   ├── uvu_bot.json         # UVU FAQ (50 examples)
│   └── cs_assistant.json    # CS concepts (50 examples)
├── examples/
│   └── load_hf_dataset.py   # How to use real HuggingFace datasets
├── requirements.txt         # Python dependencies
└── README.md
```

---

## How It Works

This project follows a simple three-step flow:

```
Edit config.yaml  -->  Run finetune.py  -->  Run app.py
```

1. **Edit `config.yaml`** -- This single file controls everything: which base model to fine-tune, which dataset to train on, training method (LoRA or full), number of epochs, batch size, and more. You rarely need to touch any Python code.

2. **Run `finetune.py`** -- This script reads your config, downloads the base model from HuggingFace, loads your dataset, tokenizes it, and trains the model. It also runs a **baseline test** (before training) and a **post-training test** (after training) so you can see the improvement. Each run is saved to the `experiments/` folder for comparison.

3. **Run `app.py`** -- This script loads your fine-tuned model and launches a Gradio web interface. You get an interactive chatbot with adjustable temperature and max token sliders, plus a shareable public link.

4. **Compare experiments** -- Run multiple experiments with different models, datasets, or settings. Each run is logged automatically. Use the comparison cell in `notebook.ipynb` (Step 7) to view all results side by side.

---

## Three Example Projects

The project comes with three ready-to-use datasets. Each one creates a different kind of chatbot. To try any of them, you only need to change **two lines** in `config.yaml`.

### 1. General Q&A Bot

A chatbot that answers general knowledge questions about science, history, geography, and more.

**Config changes** (these are the defaults):
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset:
  path: "datasets/qa_bot.json"
```

**Example interaction:**
```
You:       What is photosynthesis?
Assistant: Photosynthesis is the process by which green plants convert
           sunlight, water, and carbon dioxide into glucose and oxygen.
           It takes place in the chloroplasts of plant cells.
```

### 2. UVU FAQ Bot

A chatbot that answers frequently asked questions about Utah Valley University -- admissions, campus life, programs, and more.

**Config changes:**
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset:
  path: "datasets/uvu_bot.json"
```

**Example interaction:**
```
You:       How do I apply to UVU?
Assistant: You can apply to UVU online through the university's admissions
           website at uvu.edu/admissions. The application requires personal
           information, educational history, and a non-refundable application
           fee. UVU has an open-enrollment policy for most undergraduate
           programs.
```

### 3. CS Teaching Assistant

A chatbot that explains computer science concepts -- data structures, algorithms, programming fundamentals, and more.

**Config changes:**
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset:
  path: "datasets/cs_assistant.json"
```

**Example interaction:**
```
You:       What is a linked list?
Assistant: A linked list is a linear data structure where each element (node)
           contains data and a pointer to the next node in the sequence.
           Unlike arrays, linked lists do not store elements contiguously in
           memory, which makes insertions and deletions efficient at O(1) if
           you have a reference to the node.
```

---

## Using Real-World Data

### HuggingFace Datasets

You can fine-tune on any dataset from the [HuggingFace Hub](https://huggingface.co/datasets). The project supports this out of the box -- just update `config.yaml`:

```yaml
dataset:
  # Comment out the local path:
  # path: "datasets/qa_bot.json"

  # Use a HuggingFace dataset instead:
  hf_name: "databricks/databricks-dolly-15k"
  hf_split: "train[:500]"               # Use a subset to keep training fast
  hf_instruction_col: "instruction"      # Column containing the input/question
  hf_response_col: "response"            # Column containing the desired output
```

For a detailed walkthrough with multiple examples (Dolly, SQuAD, and more), see `examples/load_hf_dataset.py`. You can run it directly:

```bash
python examples/load_hf_dataset.py
```

### Creating Your Own Dataset

Create a JSON file with this format and save it in the `datasets/` folder:

```json
[
  {
    "instruction": "Your question or prompt here",
    "response": "The ideal answer you want the model to learn"
  },
  {
    "instruction": "Another question...",
    "response": "Another ideal answer..."
  }
]
```

Then point to it in `config.yaml`:

```yaml
dataset:
  path: "datasets/my_custom_data.json"
```

### Tips for Good Training Data

- **Start with at least 50-100 examples.** More data generally means better results, but even 50 well-crafted examples can produce noticeable improvements.
- **Be consistent in tone and format.** If you want your bot to respond in a friendly, concise style, make sure all your examples follow that pattern.
- **Cover diverse topics within your domain.** Do not repeat the same question with slight variations -- spread out across the range of things you want the bot to handle.
- **Keep responses clear and accurate.** The model will learn to mimic whatever you give it, including mistakes.

---

## Supported Models

All of these models can be used by changing the `model.name` field in `config.yaml`.

| Model | HuggingFace ID | Size | Free Colab? |
|-------|---------------|------|-------------|
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~700MB | Yes |
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | ~400MB | Yes |
| SmolLM2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Yes |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | 1B | Yes |
| Gemma 4 E2B | `google/gemma-4-e2b` | ~2B | Yes |
| Gemma 4 E4B | `google/gemma-4-e4b` | ~4B | Yes (LoRA) |
| Phi-3.5 Mini | `microsoft/Phi-3.5-mini-instruct` | 3.8B | Yes (LoRA) |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Yes |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | LoRA only |

**Recommendation:** Start with **TinyLlama** or **Qwen3 0.6B**. They are small enough to train quickly on free Colab hardware and still produce good results. Once you are comfortable, try a larger model for better quality.

> **Note:** Some models (like Llama 3.2 and Mistral) may require you to accept a license agreement on HuggingFace and log in with `huggingface-cli login` before downloading.

---

## Fine-Tuning Methods: LoRA, QLoRA, and Full

This project supports different fine-tuning approaches. Understanding the difference is important for choosing the right method for your hardware and goals.

### What is Fine-Tuning?

> **Fine-tuning** takes a pre-trained model (one that already understands language) and trains it further on your specific dataset. Instead of learning language from scratch, the model learns to specialize in your task — like teaching a chef a new recipe instead of teaching them how to cook from zero.

### LoRA (Low-Rank Adaptation)

**LoRA** is a *technique* (not a framework or library) introduced by [Microsoft Research in 2021](https://arxiv.org/abs/2106.09685). It freezes the original model weights and inserts small, trainable matrices into the model's attention layers. Instead of updating billions of parameters, you only train a small fraction (typically 1-5%).

> **How it works:** Imagine the model's knowledge as a huge book. Instead of rewriting the entire book (full fine-tuning), LoRA adds small sticky notes in key places (attention layers) that adjust the model's behavior. The original book stays intact — you only write the sticky notes.

**Read more:** [LoRA: Low-Rank Adaptation of Large Language Models (paper)](https://arxiv.org/abs/2106.09685) | [HuggingFace PEFT docs](https://huggingface.co/docs/peft)

### QLoRA (Quantized LoRA)

**QLoRA** combines LoRA with [quantization](https://huggingface.co/docs/bitsandbytes) — it first compresses the base model to use less memory (4-bit instead of 16-bit), then applies LoRA on top. This lets you fine-tune larger models on less hardware.

> **When to use:** If you want to fine-tune a 7B model (like Mistral) on free Colab, QLoRA makes it possible by reducing the base model's memory footprint by ~4x.

**Read more:** [QLoRA: Efficient Finetuning of Quantized LLMs (paper)](https://arxiv.org/abs/2305.14314) | [bitsandbytes library](https://github.com/bitsandbytes-foundation/bitsandbytes)

### Full Fine-Tuning

**Full fine-tuning** updates *all* parameters in the model. It can produce the best results but requires significantly more GPU memory and time.

> **When to use:** Only if you have access to a powerful GPU (16GB+ VRAM) and want maximum quality. For most student projects, LoRA gives you 90-95% of the quality at a fraction of the cost.

### Comparison Table

You can switch between methods in `config.yaml`:

```yaml
training:
  method: "lora"    # "lora" (recommended) or "full"
```

| | LoRA | QLoRA | Full Fine-Tuning |
|---|---|---|---|
| **What it does** | Trains small adapter layers | LoRA + quantized base model | Trains all parameters |
| **GPU Memory** | ~4-8 GB | ~2-4 GB | 16+ GB |
| **Training Speed** | Fast | Fast | Slow |
| **Free Colab?** | Yes | Yes | Maybe (small models only) |
| **Quality** | Very Good | Good | Best |
| **Best for** | Most use cases | Large models on limited hardware | Maximum quality |
| **Recommended** | Yes (default) | Advanced users | Advanced users |

### Key Terminology

| Term | What it is |
|---|---|
| **LoRA** | A technique/method — the idea of inserting small trainable matrices |
| **QLoRA** | A variant of LoRA that adds quantization to save more memory |
| **PEFT** | The [Python library](https://github.com/huggingface/peft) by HuggingFace that implements LoRA, QLoRA, and other techniques |
| **bitsandbytes** | The [library](https://github.com/bitsandbytes-foundation/bitsandbytes) that handles quantization (used by QLoRA) |

> **WARNING:** Full fine-tuning requires significantly more GPU memory than LoRA. For a 1B model, expect to need 8-16GB of VRAM. For 3B+ models, you likely need 24GB+. If you run out of memory, switch back to `method: "lora"` in `config.yaml`. Free Google Colab GPUs (T4, ~15GB VRAM) can handle LoRA for all models listed above, but full fine-tuning may only work for the smallest models.

---

## Portfolio Tips

### Get a Shareable Link

The app already launches with `share=True`, so when you run `python app.py`, Gradio will print a public URL like:

```
Running on public URL: https://xxxxx.gradio.live
```

Anyone with that link can try your chatbot. The link stays active for about 72 hours.

### Host Permanently on HuggingFace Spaces

If you want a permanent link (great for resumes and portfolios):

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Create a new Space (select "Gradio" as the SDK)
3. Upload your project files (`app.py`, `config.yaml`, `data_utils.py`, `requirements.txt`, and your `output/` folder)
4. The Space will build and deploy your chatbot automatically

### Resume / Portfolio Description

Here is an example you can adapt for your resume or portfolio:

> *Built an NLP chatbot by fine-tuning a pre-trained language model (TinyLlama 1.1B) on a custom Q&A dataset using LoRA. Deployed the model as an interactive web application using Gradio. Technologies: Python, PyTorch, HuggingFace Transformers, PEFT, Gradio.*

### Record a Demo

A short screen recording (30-60 seconds) of you chatting with your bot makes a great portfolio addition. Show the chatbot answering a few questions that demonstrate what it learned from your dataset. Most operating systems have a built-in screen recorder (QuickTime on Mac, Snipping Tool on Windows, or just use a browser extension).

---

## Troubleshooting

**Out of memory (OOM) error during training**
Your model is too large for your GPU. Try one of these:
- Switch to `method: "lora"` in `config.yaml` (if you are using full fine-tuning)
- Use a smaller model (TinyLlama or Qwen3 0.6B)
- Reduce `batch_size` to 1 or 2 in `config.yaml`

**"CUDA not available" message**
This means PyTorch does not detect a GPU. Training will still work on CPU, but it will be much slower. If you are on Colab, make sure you selected a GPU runtime: go to **Runtime > Change runtime type > T4 GPU**.

**Model not found / 401 error**
Double-check the model name in `config.yaml` for typos. Some models (Llama, Mistral) are gated and require you to:
1. Accept the license on the model's HuggingFace page
2. Log in with `huggingface-cli login` using an access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Tokenizer warnings ("Special tokens have been added...")**
These warnings are safe to ignore for most models. They appear because we set the padding token to the EOS token, which is standard practice for models that do not come with a dedicated pad token.

**"No fine-tuned model found" when running app.py**
You need to run `finetune.py` first. The app loads the trained model from the `output/` directory, which does not exist until training completes.

**Gradio link not working**
If the public Gradio link does not load, your network may be blocking outbound connections. Try running locally without the `share=True` flag, or check your firewall settings.
