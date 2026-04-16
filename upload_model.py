"""
upload_model.py — Upload Your Fine-Tuned Model to HuggingFace Hub
==================================================================
This script uploads your fine-tuned model to your HuggingFace account
so you can use it in a HuggingFace Space or share it with others.

Before running this script:
  1. Create a free account at https://huggingface.co
  2. Create an access token at https://huggingface.co/settings/tokens
     (select "Write" permission)
  3. Log in:  python -c "from huggingface_hub import login; login()"

Usage:
  python upload_model.py                          # Interactive mode
  python upload_model.py --username YOU --name my-qa-bot   # Non-interactive
  python upload_model.py --dry-run                # Check without uploading
"""

import os
import sys
import json
import yaml


def main():
    print("=" * 60)
    print("  UPLOAD FINE-TUNED MODEL TO HUGGINGFACE HUB")
    print("=" * 60)

    # ========================================================================
    # STEP 0: Parse command-line arguments
    # ========================================================================
    args = sys.argv[1:]
    cli_username = None
    cli_model_name = None
    hf_username = None
    model_name = None
    dry_run = "--dry-run" in args

    for i, arg in enumerate(args):
        if arg == "--username" and i + 1 < len(args):
            cli_username = args[i + 1]
        elif arg == "--name" and i + 1 < len(args):
            cli_model_name = args[i + 1]

    hf_username = cli_username
    model_name = cli_model_name

    # ========================================================================
    # STEP 1: Check that a fine-tuned model exists
    # ========================================================================
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["training"]["output_dir"]

    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print("\n  No fine-tuned model found!")
        print(f"  Expected output directory: {output_dir}")
        print("  Run finetune.py first.")
        return

    # Load training info for the model card
    info_path = os.path.join(output_dir, "training_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            training_info = json.load(f)
    else:
        training_info = {}

    print(f"\n  Found model in: {output_dir}")
    print(f"  Base model: {training_info.get('model_name', 'unknown')}")
    print(f"  Method: {training_info.get('method', 'unknown')}")

    # ========================================================================
    # STEP 2: Get user's HuggingFace details
    # ========================================================================
    if not hf_username:
        print("\n  Enter your HuggingFace details:")
        hf_username = input("  Your HuggingFace username: ").strip()
    if not model_name:
        model_name = input("  Name for your model (e.g., my-qa-bot): ").strip()

    if not hf_username or not model_name:
        print("  Username and model name are required!")
        return

    repo_id = f"{hf_username}/{model_name}"
    print(f"\n  Will upload to: https://huggingface.co/{repo_id}")

    if dry_run:
        print("\n  --dry-run: Skipping actual upload. Everything looks ready!")
        print(f"  To upload for real, run:")
        print(f"    python upload_model.py --username {hf_username} --name {model_name}")
        return

    # ========================================================================
    # STEP 3: Upload the model
    # ========================================================================
    # Only skip confirmation when both --username and --name were given on the CLI
    if not (cli_username and cli_model_name):
        confirm = input("  Continue? (y/n): ").strip().lower()
        if confirm != "y":
            print("  Upload cancelled.")
            return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\n  huggingface_hub not installed. Run:")
        print("    pip install huggingface_hub")
        return

    print("\n  Uploading... (this may take a few minutes)")

    api = HfApi()

    # Create a model card with training details
    model_card = f"""---
tags:
- fine-tuned
- nlp
- chatbot
---

# {model_name}

Fine-tuned from **{training_info.get('model_name', 'unknown')}** using **{training_info.get('method', 'LoRA')}**.

## Training Details
- **Base model:** {training_info.get('model_name', 'unknown')}
- **Dataset:** {training_info.get('dataset_path', 'unknown')}
- **Method:** {training_info.get('method', 'unknown')}
- **Epochs:** {training_info.get('epochs', 'unknown')}
- **Final loss:** {training_info.get('final_loss', 'unknown')}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
{"from peft import PeftModel" if training_info.get('method', 'lora') == 'lora' else ""}

# Load
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
{"base_model = AutoModelForCausalLM.from_pretrained('" + training_info.get('model_name', 'unknown') + "')" if training_info.get('method', 'lora') == 'lora' else "model = AutoModelForCausalLM.from_pretrained('" + repo_id + "')"}
{"model = PeftModel.from_pretrained(base_model, '" + repo_id + "')" if training_info.get('method', 'lora') == 'lora' else ""}
```
"""

    # Write model card to output dir temporarily
    card_path = os.path.join(output_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(model_card)

    # Create the repo if it doesn't exist, then upload
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print()
    print("=" * 60)
    print("  UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"  Model page: https://huggingface.co/{repo_id}")
    print()
    print("  Next steps:")
    print("  1. Visit the link above to see your model")
    print("  2. Create a HuggingFace Space to deploy your chatbot")
    print("     (see GUIDE.md for instructions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
