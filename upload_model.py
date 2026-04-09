"""
upload_model.py — Upload Your Fine-Tuned Model to HuggingFace Hub
==================================================================
This script uploads your fine-tuned model to your HuggingFace account
so you can use it in a HuggingFace Space or share it with others.

Before running this script:
  1. Create a free account at https://huggingface.co
  2. Create an access token at https://huggingface.co/settings/tokens
     (select "Write" permission)
  3. Log in:  huggingface-cli login

Usage:
  python upload_model.py
"""

import os
import json
import yaml


def main():
    print("=" * 60)
    print("  UPLOAD FINE-TUNED MODEL TO HUGGINGFACE HUB")
    print("=" * 60)

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
    print("\n  Enter your HuggingFace details:")
    hf_username = input("  Your HuggingFace username: ").strip()
    model_name = input("  Name for your model (e.g., my-qa-bot): ").strip()

    if not hf_username or not model_name:
        print("  Username and model name are required!")
        return

    repo_id = f"{hf_username}/{model_name}"
    print(f"\n  Will upload to: https://huggingface.co/{repo_id}")

    # ========================================================================
    # STEP 3: Upload the model
    # ========================================================================
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
from peft import PeftModel

# Load
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
base_model = AutoModelForCausalLM.from_pretrained("{training_info.get('model_name', 'unknown')}")
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```
"""

    # Write model card to output dir temporarily
    card_path = os.path.join(output_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(model_card)

    # Upload the entire output directory
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        create_repo=True,
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
