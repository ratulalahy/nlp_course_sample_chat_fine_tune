"""
app.py — Gradio Chat Interface
================================
This script launches a web-based chatbot using your fine-tuned model.

Usage:
  python app.py

The app loads the fine-tuned model from the output directory and
creates a shareable chat interface — perfect for your portfolio!

Portfolio tip: Run with share=True to get a public URL you can share!
"""

# =============================================================================
# STEP 1: IMPORTS
# We need yaml to read our config, torch for the model, gradio for the web UI,
# transformers for the tokenizer/model, and peft for LoRA adapters.
# =============================================================================
import os
import sys
import json
import yaml
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =============================================================================
# STEP 2: LOAD CONFIGURATION AND TRAINING METADATA
# We read the same config.yaml used during training, plus the metadata file
# that finetune.py saved so we know what model/dataset/method was used.
# =============================================================================

# Load the project config (model name, output directory, etc.)
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config["training"]["output_dir"]  # e.g. "output/"

# --- Check that the fine-tuned model actually exists ---
# If you haven't run finetune.py yet, there's nothing to load!
if not os.path.exists(output_dir) or not os.listdir(output_dir):
    print("=" * 60)
    print("No fine-tuned model found! Run finetune.py first.")
    print(f"  Expected output directory: {output_dir}")
    print("=" * 60)
    sys.exit(1)

# Load training metadata so we can display it in the UI
# finetune.py saves this file with info about the training run.
info_path = os.path.join(output_dir, "training_info.json")
if os.path.exists(info_path):
    with open(info_path, "r") as f:
        training_info = json.load(f)
else:
    # Fallback defaults if the file is missing (shouldn't happen normally)
    training_info = {
        "model_name": config["model"]["name"],
        "dataset_path": config.get("dataset", {}).get("path", "unknown"),
        "method": config["training"]["method"],
        "epochs": config["training"]["epochs"],
    }

# Pick the best available device: GPU if you have one, otherwise CPU.
# On Google Colab with a T4 GPU, this will select "cuda".
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# STEP 3: LOAD THE FINE-TUNED MODEL
# How we load depends on the training method:
#   - LoRA: load the original base model, then attach the LoRA adapter on top
#   - Full: the entire model was saved to output_dir, so load it directly
# =============================================================================

method = training_info.get("method", "lora")
base_model_name = training_info.get("model_name", config["model"]["name"])

# Load the tokenizer (always saved to output_dir during training)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

if method == "lora":
    # For LoRA, we first load the original base model from HuggingFace,
    # then layer the small LoRA adapter weights on top.
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.float16, device_map=device
    )
    # PeftModel.from_pretrained merges the adapter into the base model
    print(f"Loading LoRA adapter from: {output_dir}")
    model = PeftModel.from_pretrained(base_model, output_dir)
else:
    # For full fine-tuning, the whole model lives in output_dir
    print(f"Loading fully fine-tuned model from: {output_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        output_dir, dtype=torch.float16, device_map=device
    )

# Switch to evaluation mode (disables dropout, etc.)
model.eval()
print("Model loaded successfully!")

# =============================================================================
# STEP 4: DEFINE THE CHAT FUNCTION
# This is the core logic: take the user's message (plus conversation history),
# build a prompt the model understands, generate a response, and return it.
# =============================================================================


def chat(message, history, temperature, max_tokens):
    """
    Generate a response to the user's message.

    Args:
        message:     The user's new message (string)
        history:     Previous conversation as list of {"role": ..., "content": ...} dicts
        temperature: Controls randomness (0.1 = focused, 1.5 = creative)
        max_tokens:  Maximum number of tokens to generate in the response
    """
    # --- Build the prompt from conversation history + new message ---
    # We use a simple format the model was trained on:
    #   ### User: <question>
    #   ### Assistant: <answer>
    # For multi-turn conversations, we prepend all previous turns.
    prompt = ""
    for turn in history:
        if turn["role"] == "user":
            prompt += f"### User: {turn['content']}\n"
        elif turn["role"] == "assistant":
            prompt += f"### Assistant: {turn['content']}\n"
    # Append the new user message and prompt the assistant to respond
    prompt += f"### User: {message}\n### Assistant:"

    # --- Tokenize the prompt and send it to the model's device ---
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # --- Generate a response ---
    # We use torch.no_grad() because we're only doing inference (no training).
    # This saves memory and speeds things up.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=True,          # Enable sampling (needed for temperature)
            top_p=0.9,               # Nucleus sampling for quality
            repetition_penalty=1.1,  # Discourage repetitive text
            pad_token_id=tokenizer.eos_token_id,  # Avoid padding warnings
        )

    # --- Decode and extract only the NEW assistant response ---
    # The model returns the full sequence (prompt + response), so we
    # slice off the prompt tokens to get just the generated part.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


# =============================================================================
# STEP 5: BUILD AND LAUNCH THE GRADIO INTERFACE
# Gradio makes it dead-simple to create a web-based chat UI.
# ChatInterface handles the conversation loop automatically.
# =============================================================================

# Build a description string showing training details
description = (
    f"**Model:** {training_info.get('model_name', 'N/A')}\n\n"
    f"**Dataset:** {training_info.get('dataset_path', 'N/A')}\n\n"
    f"**Method:** {training_info.get('method', 'N/A')} "
    f"| **Epochs:** {training_info.get('epochs', 'N/A')}"
)

# Create the chat interface with adjustable generation settings
demo = gr.ChatInterface(
    fn=chat,
    type="messages",  # Modern format: list of {"role": ..., "content": ...} dicts
    title="Fine-Tuned NLP Chatbot",
    description=description,
    # Example prompts so users can try the bot right away
    # Each example is a list: [message, temperature, max_tokens]
    # because we have additional_inputs (sliders) defined below.
    examples=[
        ["What is photosynthesis?"],
        ["How far is the Moon from Earth?"],
        ["Explain what machine learning is."],
        ["What causes earthquakes?"],
    ],
    # Sliders let users tweak generation on the fly
    additional_inputs=[
        gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature",
                  info="Lower = more focused, higher = more creative"),
        gr.Slider(50, 500, value=200, step=50, label="Max Tokens",
                  info="Maximum length of the generated response"),
    ],
)

# Launch the app!
# share=True creates a temporary public URL (great for portfolio demos).
# Anyone with the link can try your chatbot for 72 hours.
demo.launch(share=True)
