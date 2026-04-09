"""
load_hf_dataset.py — How to Use Real-World Datasets from HuggingFace
=====================================================================
This script shows you how to:
  1. Load any dataset from HuggingFace Hub
  2. Explore what's inside a dataset
  3. Map its columns to our instruction/response format
  4. Save a subset locally for fine-tuning

You can run this script directly:
  python examples/load_hf_dataset.py

After running, check the datasets/ folder for the downloaded data!
"""

from datasets import load_dataset
import json
import os

# ============================================
# EXAMPLE 1: Dolly Dataset (Instruction-Following)
# ============================================
# Databricks Dolly is a great dataset for instruction-following chatbots.
# It contains ~15,000 instruction/response pairs across many categories.
#
# Let's load it, explore it, and save a subset in our format.


def load_dolly_example():
    """Load the Dolly dataset and convert to our format."""

    print("=" * 60)
    print("EXAMPLE 1: Loading Dolly Dataset")
    print("=" * 60)

    # Load the dataset from HuggingFace Hub
    # The first time you run this, it will download the dataset.
    # After that, it uses a cached copy so it's fast.
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Let's see what columns it has
    # Every HuggingFace dataset has column_names and a length
    print(f"\nColumns: {dataset.column_names}")
    print(f"Number of examples: {len(dataset)}")
    print(f"\nFirst example:")
    print(f"  Instruction: {dataset[0]['instruction'][:100]}...")
    print(f"  Response: {dataset[0]['response'][:100]}...")

    # Convert to our simple format (instruction/response pairs)
    # We only take the first 200 examples to keep it small.
    # For real fine-tuning you might use more, but 200 is enough
    # to test your pipeline.
    converted = []
    for example in dataset.select(range(min(200, len(dataset)))):
        converted.append({
            "instruction": example["instruction"],
            "response": example["response"]
        })

    # Save to a JSON file in our datasets folder
    output_path = os.path.join("datasets", "dolly_subset.json")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"\nSaved {len(converted)} examples to {output_path}")


# ============================================
# EXAMPLE 2: SQuAD Dataset (Question Answering)
# ============================================
# SQuAD is a popular question-answering dataset.
# It has a different format (context + question + answers), so we need
# to map its columns to our instruction/response format.
#
# This is a common pattern: most real-world datasets have different
# column names than what we use. You just need to figure out which
# column maps to "instruction" and which maps to "response".


def load_squad_example():
    """Load SQuAD and map its columns to our format."""

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Loading SQuAD Dataset")
    print("=" * 60)

    # Load a small subset of SQuAD
    # The "split" argument lets you grab just part of the data.
    # "train[:200]" means "first 200 examples from the training set".
    dataset = load_dataset("rajpurkar/squad", split="train[:200]")

    print(f"\nColumns: {dataset.column_names}")
    print(f"Number of examples: {len(dataset)}")

    # SQuAD has: context, question, answers
    # We need to map: question -> instruction, answers -> response
    #
    # Note: "answers" in SQuAD is a dict with a "text" list inside it.
    # This kind of nested structure is common in HuggingFace datasets.
    print(f"\nFirst example:")
    print(f"  Question: {dataset[0]['question']}")
    print(f"  Answer: {dataset[0]['answers']['text'][0]}")

    # Convert to our format
    # Notice how we map different column names to instruction/response.
    # This is the key skill: figuring out which columns in the source
    # dataset correspond to the fields your fine-tuning pipeline expects.
    converted = []
    for example in dataset:
        if example["answers"]["text"]:  # Skip if no answer
            converted.append({
                "instruction": example["question"],
                "response": example["answers"]["text"][0]
            })

    output_path = os.path.join("datasets", "squad_subset.json")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"\nSaved {len(converted)} examples to {output_path}")


# ============================================
# EXAMPLE 3: Browsing HuggingFace Hub
# ============================================
# Here are some other great datasets you can try!
# You can browse thousands more at https://huggingface.co/datasets


def show_popular_datasets():
    """Print a list of popular datasets for chatbot fine-tuning."""

    print("\n" + "=" * 60)
    print("POPULAR DATASETS FOR CHATBOT FINE-TUNING")
    print("=" * 60)

    datasets_list = [
        ("databricks/databricks-dolly-15k", "15K instruction-following examples"),
        ("rajpurkar/squad", "100K+ question-answering examples"),
        ("Open-Orca/OpenOrca", "Large instruction dataset"),
        ("yahma/alpaca-cleaned", "52K cleaned instruction examples"),
        ("timdettmers/openassistant-guanaco", "10K chat conversations"),
    ]

    print("\nDataset Name                          | Description")
    print("-" * 70)
    for name, desc in datasets_list:
        print(f"  {name:<38} | {desc}")

    print("\nTo use any of these, update config.yaml:")
    print("  1. Comment out the 'path' line under dataset")
    print("  2. Uncomment and set 'hf_name' to the dataset name")
    print("  3. Set 'hf_instruction_col' and 'hf_response_col' to match the columns")
    print("\nOr use this script as a starting point to download and convert them!")


# ============================================
# TIP: Creating Your Own Dataset
# ============================================
# If none of the existing datasets fit your needs, you can always
# create your own! Here's how.


def show_custom_dataset_tip():
    """Show how to create a custom dataset from scratch."""

    print("\n" + "=" * 60)
    print("TIP: CREATING YOUR OWN DATASET")
    print("=" * 60)

    print("""
To create your own dataset for fine-tuning, just make a JSON file
with this format:

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

Tips for good datasets:
  - Start with at least 50-100 examples
  - Be consistent in tone and format
  - Cover diverse topics within your domain
  - Keep responses clear and accurate
  - Save as a .json file in the datasets/ folder

Then just update config.yaml to point to your file:
  dataset:
    path: "datasets/my_custom_data.json"
""")


# ============================================
# RUN ALL EXAMPLES
# ============================================
if __name__ == "__main__":
    print("HuggingFace Dataset Loading Examples")
    print("=" * 60)

    # Make sure datasets folder exists
    os.makedirs("datasets", exist_ok=True)

    # Run each example
    load_dolly_example()
    load_squad_example()
    show_popular_datasets()
    show_custom_dataset_tip()

    print("\nDone! Check the datasets/ folder for downloaded data.")
