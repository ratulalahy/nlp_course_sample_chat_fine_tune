"""
finetune.py -- Fine-Tune a Small Language Model
================================================
This script fine-tunes a language model using either:
  - LoRA (recommended, memory-efficient)
  - Full fine-tuning (WARNING: requires more GPU memory)

Usage:
  python finetune.py

All settings come from config.yaml -- edit that file to change
the model, dataset, or training parameters.
"""

# ============================================================================
# STEP 1: Import Libraries
# ============================================================================
# We need a handful of libraries:
#   - torch: PyTorch, the deep learning framework that powers everything
#   - transformers: HuggingFace's library for loading pre-trained models
#   - peft: Parameter-Efficient Fine-Tuning (for LoRA)
#   - data_utils: our helper module for loading configs and datasets
#   - time / json / os: Python built-ins for timing, saving metadata, file paths

import time
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,       # Loads any causal language model (GPT-style)
    AutoTokenizer,              # Loads the matching tokenizer for any model
    TrainingArguments,          # Configures HOW training runs (epochs, batch size, etc.)
    Trainer,                    # The actual training loop from HuggingFace
)
from peft import LoraConfig, get_peft_model, TaskType

from data_utils import load_config, load_dataset, format_for_training


def main():
    """
    Main fine-tuning pipeline. Runs all steps in order:
    load config -> load data -> load model -> tokenize -> train -> save.
    """

    print("=" * 60)
    print("  FINE-TUNING SCRIPT")
    print("  Reading all settings from config.yaml")
    print("=" * 60)

    # ========================================================================
    # STEP 2: Load Configuration
    # ========================================================================
    # Everything is controlled by config.yaml: which model to use, which
    # dataset to train on, how many epochs, learning rate, etc.
    # We load it once and pass pieces to each step below.

    config = load_config("config.yaml")

    # Pull out the key settings so we can reference them easily later.
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    method = config["training"]["method"]           # "lora" or "full"
    output_dir = config["training"]["output_dir"]

    print(f"\n  Model:   {model_name}")
    print(f"  Method:  {method}")
    print(f"  Epochs:  {config['training']['epochs']}")
    print(f"  Output:  {output_dir}\n")

    # ========================================================================
    # STEP 3: Load the Dataset
    # ========================================================================
    # data_utils.load_dataset() reads our config and returns a list of
    # {"instruction": "...", "response": "..."} dictionaries.
    # It handles both local JSON files and HuggingFace Hub datasets.

    examples = load_dataset(config)

    print(f"  Loaded {len(examples)} training examples")
    print(f"  First example:")
    print(f"    Instruction: {examples[0]['instruction'][:80]}...")
    print(f"    Response:    {examples[0]['response'][:80]}...")
    print()

    # ========================================================================
    # STEP 4: Load the Base Model and Tokenizer
    # ========================================================================
    # We download the pre-trained model from HuggingFace Hub. This is the
    # "starting point" -- a model that already knows language but hasn't been
    # specialized for our task yet. Fine-tuning will teach it our data.

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Many models (like LLaMA, TinyLlama) don't have a padding token set.
    # Without a padding token, the Trainer will crash when it tries to batch
    # sequences of different lengths. We fix this by reusing the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  (Set pad_token = eos_token -- common fix for many models)")

    print("  Loading model... (this may take a minute the first time)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,         # Use half-precision to save memory
        device_map="auto",                 # Automatically use GPU if available
    )

    # Print model size so students can see how big it is.
    # "Parameters" are the numbers the model learned during pre-training.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {model_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (approx): {total_params * 2 / 1e9:.2f} GB (float16)")
    print()

    # ========================================================================
    # STEP 5: Prepare Data for Training
    # ========================================================================
    # We need to convert our human-readable instruction/response pairs into
    # numbers (token IDs) that the model understands. This is called
    # "tokenization". format_for_training() does this for us and returns
    # a HuggingFace Dataset object ready for the Trainer.

    # ========================================================================
    # STEP 5a: Baseline Test (BEFORE Fine-Tuning)
    # ========================================================================
    # Let's see how the BASE model (before any fine-tuning) responds to a
    # few questions from our dataset. This gives us a "before" snapshot so
    # students can compare it with the "after" results.
    # This is saved to the experiment log so you can see the improvement.

    print("  Running baseline test (before fine-tuning)...")
    baseline_questions = [ex["instruction"] for ex in examples[:3]]
    baseline_results = []

    model.eval()
    for question in baseline_questions:
        prompt = f"### User: {question}\n### Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        baseline_results.append({"question": question, "response": response})
        print(f"    Q: {question[:60]}...")
        print(f"    A: {response[:80]}...")
        print()
    model.train()

    print("  Tokenizing dataset...")
    train_dataset = format_for_training(examples, tokenizer, max_length)
    print(f"  Dataset ready: {len(train_dataset)} tokenized examples")
    print()

    # ========================================================================
    # STEP 6: Set Up Fine-Tuning Method
    # ========================================================================
    # Two options:
    #   "lora" -- Only train small adapter layers (~1-5% of parameters).
    #             Fast, memory-efficient. Works on free Google Colab.
    #   "full" -- Train ALL parameters. Better results but needs much more
    #             GPU memory (16GB+). NOT recommended for free Colab.

    if method == "lora":
        print("  Setting up LoRA (Low-Rank Adaptation)...")

        # LoRA inserts small trainable matrices into the model's attention layers.
        # We configure: rank (r), scaling (alpha), dropout, and which layers to target.
        #
        # target_modules: which layers to apply LoRA to.
        # If set to null/None in config, we default to common attention layer names
        # that work across most model architectures (LLaMA, Gemma, Qwen, Mistral, etc.)
        target_modules = config["lora"].get("target_modules")
        if target_modules is None:
            # These layer names cover most popular model architectures:
            #   q_proj, v_proj  -> LLaMA, Mistral, Qwen
            #   q_proj, v_proj  -> Gemma (also uses these names)
            target_modules = ["q_proj", "v_proj"]
            print(f"  Auto-selecting target_modules: {target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,                       # We're training a language model
            r=config["lora"]["r"],                              # Rank: controls adapter size
            lora_alpha=config["lora"]["alpha"],                 # Scaling factor
            lora_dropout=config["lora"]["dropout"],             # Regularization
            target_modules=target_modules,                      # Which layers to apply LoRA to
        )

        # Wrap the model with LoRA adapters. This "freezes" the original model
        # weights and only makes the small LoRA layers trainable.
        model = get_peft_model(model, lora_config)

        # Show how many parameters we're actually training vs. the total.
        # This is the magic of LoRA: we train <5% of the model's parameters.
        model.print_trainable_parameters()
        print()

    elif method == "full":
        # ===================================================================
        # WARNING: Full fine-tuning updates ALL model parameters.
        # This requires MUCH more GPU memory than LoRA.
        # For a 1B model, you need at least 8-16GB of GPU memory.
        # For 3B+ models, you likely need 24GB+.
        # ===================================================================
        print("!" * 60)
        print("  WARNING: Full fine-tuning selected!")
        print("  This requires significantly more GPU memory than LoRA.")
        print("  - 1B models:  ~8-16GB GPU memory needed")
        print("  - 3B models:  ~16-24GB GPU memory needed")
        print("  - 7B models:  ~32GB+ GPU memory needed")
        print("  If you run out of memory, switch to method: 'lora' in config.yaml")
        print("!" * 60)
        print()
    else:
        raise ValueError(
            f"Unknown training method: '{method}'. "
            f"Use 'lora' or 'full' in config.yaml."
        )

    # ========================================================================
    # STEP 7: Configure Training
    # ========================================================================
    # TrainingArguments tells the Trainer everything about HOW to train:
    # how many epochs, batch size, learning rate, where to save checkpoints, etc.
    # We pull all of these values from config.yaml so you can tweak them easily.

    training_args = TrainingArguments(
        output_dir=output_dir,                                  # Where to save checkpoints
        num_train_epochs=config["training"]["epochs"],          # How many passes over the data
        per_device_train_batch_size=config["training"]["batch_size"],  # Samples per step
        learning_rate=config["training"]["learning_rate"],      # Step size for optimizer
        logging_steps=config["training"]["logging_steps"],      # Print loss every N steps
        save_strategy="epoch",                                  # Save a checkpoint each epoch
        fp16=torch.cuda.is_available(),                         # Use half-precision on GPU
        report_to="none",                                       # Don't log to wandb/tensorboard
        remove_unused_columns=False,                            # Keep all dataset columns
    )

    # ========================================================================
    # STEP 8: Train the Model!
    # ========================================================================
    # This is where the actual learning happens. The Trainer will:
    #   1. Loop over the dataset for the specified number of epochs
    #   2. Feed batches through the model
    #   3. Compute the loss (how wrong the model's predictions are)
    #   4. Update weights to reduce the loss (backpropagation)
    # We time it so you can see how long training takes.

    print("  Starting training...")
    print("  (You'll see the loss decrease over time -- that means it's learning!)")
    print()

    # Record the start time so we can report total duration later.
    start_time = time.time()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # trainer.train() runs the full training loop. This is the big moment!
    train_result = trainer.train()

    # Calculate how long training took.
    training_time = time.time() - start_time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)

    print()
    print(f"  Training complete! Duration: {minutes}m {seconds}s")

    # ========================================================================
    # STEP 9: Save the Fine-Tuned Model
    # ========================================================================
    # Now we save everything needed to load the model later in app.py:
    #   - For LoRA: save the adapter weights (small, ~10-50MB)
    #   - For full: save the entire model (same size as the original)
    #   - Always: save the tokenizer so we can decode text later

    print(f"\n  Saving model to {output_dir}...")

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    if method == "lora":
        # Save only the LoRA adapter weights (not the full model).
        # This is much smaller -- typically 10-50MB instead of several GB.
        # When we load it in app.py, we'll load the base model first,
        # then apply these adapter weights on top.
        model.save_pretrained(output_dir)
        print("  Saved LoRA adapter weights")
    else:
        # Save the full fine-tuned model. This is a complete copy of the model
        # with all weights updated. It's bigger but simpler to load.
        trainer.save_model(output_dir)
        print("  Saved full fine-tuned model")

    # Always save the tokenizer alongside the model.
    # The tokenizer converts text to numbers and back -- we need it for inference.
    tokenizer.save_pretrained(output_dir)
    print("  Saved tokenizer")

    # Save a small metadata file so app.py knows what was trained.
    # This is handy for displaying info in the chat interface.
    training_info = {
        "model_name": model_name,
        "dataset_path": config["dataset"].get("path", config["dataset"].get("hf_name", "unknown")),
        "method": method,
        "epochs": config["training"]["epochs"],
        "final_loss": round(train_result.training_loss, 4),
        "training_time_seconds": round(training_time, 1),
    }

    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"  Saved training metadata to {info_path}")

    # ========================================================================
    # STEP 10: Post-Training Test (AFTER Fine-Tuning)
    # ========================================================================
    # Run the SAME questions from the baseline test on the fine-tuned model.
    # This lets students directly compare "before" vs "after" responses.

    print("\n  Running post-training test (after fine-tuning)...")
    after_results = []

    model.eval()
    for question in baseline_questions:
        prompt = f"### User: {question}\n### Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        after_results.append({"question": question, "response": response})
        print(f"    Q: {question[:60]}...")
        print(f"    A: {response[:80]}...")
        print()

    # ========================================================================
    # STEP 11: Save Experiment Log
    # ========================================================================
    # Each training run is saved to experiments/ with a timestamp.
    # This lets students compare results across different configurations:
    #   - Different models (TinyLlama vs Gemma vs Qwen)
    #   - Different datasets (QA vs UVU vs CS)
    #   - Different settings (LoRA vs full, epochs, learning rate)
    #
    # Check the experiments/ folder to see all your past runs!

    from datetime import datetime

    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    # Create a descriptive filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]  # e.g. "TinyLlama-1.1B-Chat-v1.0"
    dataset_short = config["dataset"].get("path", config["dataset"].get("hf_name", "unknown"))
    dataset_short = os.path.basename(dataset_short).replace(".json", "")
    exp_filename = f"{timestamp}_{model_short}_{dataset_short}_{method}.json"

    experiment = {
        "timestamp": timestamp,
        "config": {
            "model_name": model_name,
            "dataset": config["dataset"].get("path", config["dataset"].get("hf_name", "unknown")),
            "method": method,
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "lora_r": config["lora"]["r"] if method == "lora" else None,
            "lora_alpha": config["lora"]["alpha"] if method == "lora" else None,
        },
        "results": {
            "final_loss": round(train_result.training_loss, 4),
            "training_time_seconds": round(training_time, 1),
            "total_parameters": total_params,
        },
        "baseline_responses": baseline_results,
        "finetuned_responses": after_results,
    }

    exp_path = os.path.join(experiments_dir, exp_filename)
    with open(exp_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"  Experiment saved to {exp_path}")
    print(f"  Compare with past experiments in the experiments/ folder!")

    # ========================================================================
    # STEP 12: Print Summary
    # ========================================================================
    # Give the student a clear summary of what happened and what to do next.

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  Method:         {method}")
    print(f"  Epochs:         {config['training']['epochs']}")
    print(f"  Final Loss:     {train_result.training_loss:.4f}")
    print(f"  Training Time:  {minutes}m {seconds}s")
    print(f"  Saved To:       {output_dir}")
    print(f"  Experiment Log: {exp_path}")
    print()
    print("  NEXT STEPS:")
    print("  1. Run the chatbot:  python app.py")
    print("  2. Chat with your fine-tuned model!")
    print("  3. Share the Gradio link for your portfolio")
    print("=" * 60)


if __name__ == "__main__":
    main()
