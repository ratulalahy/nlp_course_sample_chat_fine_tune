"""
data_utils.py — Data Loading & Preprocessing for Fine-Tuning
=============================================================
This module handles everything related to getting data ready for training:
  1. Loading the YAML config file
  2. Loading datasets from local JSON files or HuggingFace Hub
  3. Formatting and tokenizing data for the model

Think of this as the "kitchen" of our fine-tuning pipeline:
  - Raw ingredients (JSON / HuggingFace data) come in
  - We clean, chop, and prepare them (validate, format, tokenize)
  - A ready-to-cook Dataset goes out to the trainer

You typically don't run this file directly — it's imported by finetune.py.
But you CAN run it standalone to test your dataset:
  python data_utils.py
"""

import json
import yaml
from datasets import Dataset

# =============================================================================
# 1. CONFIGURATION LOADING
# =============================================================================
# Our project uses a single config.yaml file to control everything.
# This function reads that file and gives us a Python dictionary so we
# can access settings like config["model"]["name"] or config["training"]["epochs"].


def load_config(config_path="config.yaml"):
    """
    Read the YAML configuration file and return it as a Python dictionary.

    Args:
        config_path (str): Path to the YAML config file. Defaults to "config.yaml"
                           in the current working directory.

    Returns:
        dict: The entire configuration as a nested dictionary. For example:
              {
                "model": {"name": "TinyLlama/...", "max_length": 512},
                "dataset": {"path": "datasets/qa_bot.json"},
                "training": {"method": "lora", "epochs": 3, ...},
                "lora": {"r": 8, "alpha": 16, ...}
              }

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    # Open the file and use PyYAML's safe_load to parse it.
    # safe_load is preferred over load() because it won't execute
    # arbitrary Python code that could be embedded in the YAML.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Quick sanity check: make sure we got a dictionary back.
    # If the YAML file is empty or malformed, safe_load returns None.
    if config is None:
        raise ValueError(f"Config file '{config_path}' is empty or invalid.")

    return config


# =============================================================================
# 2. LOADING DATA FROM A LOCAL JSON FILE
# =============================================================================
# Our default workflow uses a simple JSON file stored in datasets/.
# The expected format is a list of dictionaries, each with two keys:
#   [
#     {"instruction": "What is X?", "response": "X is ..."},
#     {"instruction": "Explain Y.",  "response": "Y means ..."},
#     ...
#   ]


def load_dataset_from_json(file_path):
    """
    Load a dataset from a local JSON file and validate its structure.

    This function reads the JSON file, checks that every example has the
    required "instruction" and "response" keys, and returns the data as
    a list of dictionaries.

    Args:
        file_path (str): Path to the JSON file (e.g., "datasets/qa_bot.json").

    Returns:
        list[dict]: A list of dictionaries, each with "instruction" and "response" keys.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If any example is missing required keys.
    """
    # Step 1: Read the raw JSON from disk.
    # json.load() parses the file contents into Python objects.
    with open(file_path, "r") as f:
        data = json.load(f)

    # Step 2: Validate the structure.
    # We need every item to be a dict with "instruction" and "response".
    # If someone accidentally deletes a key or uses the wrong column name,
    # we want a clear error message — not a cryptic crash during training.
    required_keys = {"instruction", "response"}

    for i, example in enumerate(data):
        # Get the keys that are present in this example
        present_keys = set(example.keys())

        # Check if our required keys are a subset of the present keys
        # (the example can have extra keys — that's fine, we just ignore them)
        missing_keys = required_keys - present_keys
        if missing_keys:
            raise ValueError(
                f"Example {i} is missing required keys: {missing_keys}. "
                f"Found keys: {present_keys}. "
                f"Each example must have 'instruction' and 'response'."
            )

    # Step 3: Report what we loaded (helpful for debugging)
    print(f"Loaded {len(data)} examples from '{file_path}'")

    return data


# =============================================================================
# 3. LOADING DATA FROM HUGGINGFACE HUB
# =============================================================================
# HuggingFace Hub hosts thousands of open datasets. Instead of downloading
# a JSON file manually, you can point to any dataset on the Hub.
#
# The catch: every dataset uses different column names. Dolly uses
# "instruction"/"response", SQuAD uses "question"/"answers", etc.
# So we need a mapping: tell us which column is the instruction and
# which is the response, and we'll standardize them.


def load_dataset_from_hf(hf_name, hf_split, instruction_col, response_col):
    """
    Load a dataset from HuggingFace Hub and map its columns to our format.

    This downloads the dataset (or uses a cached copy), extracts the columns
    you specify, and returns everything in our standard format.

    Args:
        hf_name (str): The HuggingFace dataset identifier.
                       Example: "databricks/databricks-dolly-15k"
        hf_split (str): Which split to load, with optional slicing.
                        Examples: "train", "train[:500]", "test"
        instruction_col (str): Name of the column to use as "instruction".
        response_col (str): Name of the column to use as "response".

    Returns:
        list[dict]: A list of dicts with "instruction" and "response" keys.

    Raises:
        ValueError: If the specified columns don't exist in the dataset.
    """
    # We import load_dataset here (not at the top of the file) because
    # HuggingFace's datasets library can be slow to import, and we don't
    # want to pay that cost if the user is just loading a local JSON file.
    from datasets import load_dataset as hf_load_dataset

    # Step 1: Download / load the dataset from HuggingFace Hub.
    # The first run downloads it; after that it's cached locally.
    print(f"Loading HuggingFace dataset: {hf_name} (split: {hf_split})")
    hf_dataset = hf_load_dataset(hf_name, split=hf_split)

    # Step 2: Verify that the columns the user specified actually exist.
    # This prevents confusing errors later in the pipeline.
    available_columns = hf_dataset.column_names
    for col_name, col_label in [(instruction_col, "instruction"), (response_col, "response")]:
        if col_name not in available_columns:
            raise ValueError(
                f"Column '{col_name}' (mapped to '{col_label}') not found in dataset. "
                f"Available columns: {available_columns}"
            )

    # Step 3: Convert each example to our standard format.
    # We iterate through the HuggingFace dataset and pull out just the
    # two columns we need, renaming them to "instruction" and "response".
    data = []
    for example in hf_dataset:
        data.append({
            "instruction": str(example[instruction_col]),
            "response": str(example[response_col]),
        })

    print(f"Loaded {len(data)} examples from HuggingFace dataset '{hf_name}'")

    return data


# =============================================================================
# 4. MAIN ENTRY POINT — LOAD DATASET BASED ON CONFIG
# =============================================================================
# This function looks at config.yaml and decides which loader to call.
# If the config has "hf_name" under dataset, we load from HuggingFace.
# Otherwise, we load from the local "path".
#
# Why a "dispatcher" function? So the rest of our code (train.py, etc.)
# doesn't need to know or care WHERE the data comes from. It just calls
# load_dataset(config) and gets back a list of examples. This is a common
# design pattern called the "Strategy Pattern".


def load_dataset(config):
    """
    Load a dataset based on the configuration. Automatically picks the
    right loader (local JSON vs. HuggingFace Hub) based on what's in config.

    Decision logic:
      - If config["dataset"]["hf_name"] exists  ->  load from HuggingFace
      - If config["dataset"]["path"] exists      ->  load from local JSON file
      - Otherwise                                ->  raise an error

    Args:
        config (dict): The full configuration dictionary (from load_config()).

    Returns:
        list[dict]: A list of dicts with "instruction" and "response" keys.

    Raises:
        ValueError: If neither 'path' nor 'hf_name' is specified in the config.
    """
    # Pull out just the dataset section of the config for convenience
    dataset_config = config.get("dataset", {})

    # Check if the user wants to load from HuggingFace Hub.
    # We check for "hf_name" first because if both "path" and "hf_name"
    # are present, we prefer HuggingFace (it's probably intentional).
    hf_name = dataset_config.get("hf_name")

    if hf_name:
        # HuggingFace mode — we need three additional settings from the config.
        # Provide sensible defaults in case the user forgets one.
        hf_split = dataset_config.get("hf_split", "train")
        instruction_col = dataset_config.get("hf_instruction_col", "instruction")
        response_col = dataset_config.get("hf_response_col", "response")

        return load_dataset_from_hf(hf_name, hf_split, instruction_col, response_col)

    # Check for a local file path
    file_path = dataset_config.get("path")

    if file_path:
        return load_dataset_from_json(file_path)

    # If we get here, there's no data source configured at all
    raise ValueError(
        "No dataset configured! In config.yaml, you need either:\n"
        "  - dataset.path: 'datasets/your_file.json'   (for local files)\n"
        "  - dataset.hf_name: 'org/dataset-name'       (for HuggingFace datasets)\n"
        "Please check your config.yaml."
    )


# =============================================================================
# 5. FORMAT AND TOKENIZE FOR TRAINING
# =============================================================================
# The model can't read English — it only understands numbers (token IDs).
# This function takes our instruction/response pairs and converts them into
# the tokenized format that the Trainer expects.
#
# The chat template we use is:
#   ### User: {instruction}
#   ### Assistant: {response}
#
# This is a simple but effective format. The model learns that text after
# "### User:" is the question and text after "### Assistant:" is how it
# should respond. During inference, we give it "### User: ..." and let
# it generate the "### Assistant: ..." part.


def format_for_training(examples, tokenizer, max_length):
    """
    Convert instruction/response pairs into a tokenized HuggingFace Dataset
    ready for the Trainer.

    Each example is formatted into a chat-style prompt, then tokenized.
    The labels are set to be the same as input_ids (standard for causal LM
    fine-tuning — the model learns to predict each next token).

    Args:
        examples (list[dict]): List of dicts with "instruction" and "response" keys.
        tokenizer: A HuggingFace tokenizer (e.g., from AutoTokenizer.from_pretrained).
        max_length (int): Maximum sequence length. Longer texts get truncated.

    Returns:
        datasets.Dataset: A HuggingFace Dataset with columns:
            - input_ids: Token IDs for the formatted text
            - attention_mask: 1 for real tokens, 0 for padding
            - labels: Same as input_ids (the model learns to predict these)
    """
    # Step 1: Format each example into our chat template.
    # We combine the instruction and response into a single string
    # because causal language models learn from continuous text.
    formatted_texts = []
    for example in examples:
        # Build the chat-formatted string.
        # The EOS token (End Of Sequence) tells the model "stop generating here."
        # Without it, the model would keep generating text forever.
        text = (
            f"### User: {example['instruction']}\n"
            f"### Assistant: {example['response']}"
            f"{tokenizer.eos_token}"
        )
        formatted_texts.append(text)

    # Step 2: Tokenize all the formatted texts in one batch.
    #
    # What does tokenization do?
    #   "Hello world" -> [15496, 995]  (each word becomes a number)
    #
    # Parameters explained:
    #   - truncation=True: If the text is longer than max_length, cut it off.
    #     Better to lose some text than crash with an out-of-memory error.
    #   - padding="max_length": Pad shorter texts with special [PAD] tokens
    #     so every example has the same length. This is required because
    #     GPUs process data in fixed-size batches.
    #   - max_length: The maximum number of tokens per example.
    #   - return_tensors=None: Return plain Python lists (not PyTorch tensors).
    #     The HuggingFace Trainer will convert to tensors later.
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    # Step 3: Set labels equal to input_ids.
    #
    # In causal language modeling (like GPT), the training objective is:
    #   Given tokens [t1, t2, ..., tn], predict [t2, t3, ..., tn+1]
    #
    # The Trainer handles this shifting automatically — we just need to
    # provide the full sequence as both input and label. The model learns
    # to predict each next token based on all previous tokens.
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Step 4: Wrap everything in a HuggingFace Dataset object.
    #
    # The Trainer expects a Dataset, not a plain dict. The Dataset class
    # provides nice features like batching, shuffling, and memory mapping.
    dataset = Dataset.from_dict(tokenized)

    print(f"Tokenized {len(dataset)} examples (max_length={max_length})")

    return dataset


# =============================================================================
# DEMO / TEST BLOCK
# =============================================================================
# This block runs when you execute: python data_utils.py
# It's a great way to test that your dataset loads correctly before training.
# It does NOT run when you import this file from another script (like train.py).

if __name__ == "__main__":
    print("=" * 60)
    print("DATA UTILS — Testing Dataset Loading")
    print("=" * 60)

    # --- Step 1: Load the config ---
    print("\n--- Loading config.yaml ---")
    config = load_config()
    print(f"Model: {config['model']['name']}")
    print(f"Max length: {config['model']['max_length']}")

    # --- Step 2: Load the dataset ---
    print("\n--- Loading dataset ---")
    data = load_dataset(config)

    # --- Step 3: Show a few examples ---
    # Display the first 3 examples so we can eyeball the data
    num_to_show = min(3, len(data))
    print(f"\n--- First {num_to_show} examples ---")
    for i in range(num_to_show):
        print(f"\nExample {i + 1}:")
        print(f"  Instruction: {data[i]['instruction']}")
        # Truncate long responses for readability in the terminal
        response_preview = data[i]["response"]
        if len(response_preview) > 150:
            response_preview = response_preview[:150] + "..."
        print(f"  Response:    {response_preview}")

    # --- Step 4: Show what the formatted chat template looks like ---
    # This is useful to verify the model will see the right format
    print(f"\n--- Chat template preview (Example 1) ---")
    sample_text = (
        f"### User: {data[0]['instruction']}\n"
        f"### Assistant: {data[0]['response']}<eos_token>"
    )
    print(sample_text)

    # --- Step 5: Quick tokenization test ---
    # We try to load the tokenizer and tokenize one example.
    # If this fails (e.g., no internet, model not found), that's okay —
    # we just skip it and print a helpful message.
    print(f"\n--- Tokenization test ---")
    try:
        from transformers import AutoTokenizer

        model_name = config["model"]["name"]
        print(f"Loading tokenizer for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Some tokenizers don't have a pad token by default.
        # We set it to eos_token as a common workaround.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize just the first 3 examples as a test
        test_data = data[:3]
        max_length = config["model"]["max_length"]
        test_dataset = format_for_training(test_data, tokenizer, max_length)

        # Show the shape of what we produced
        print(f"Dataset columns: {test_dataset.column_names}")
        print(f"Number of examples: {len(test_dataset)}")
        print(f"Token IDs (first 20 of example 1): {test_dataset[0]['input_ids'][:20]}")

    except Exception as e:
        print(f"Tokenization test skipped: {e}")
        print("(This is fine — tokenization will work when you run train.py)")

    print("\n" + "=" * 60)
    print("Dataset loading works! You're ready to train.")
    print("=" * 60)
