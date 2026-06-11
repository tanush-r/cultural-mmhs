import os
import numpy as np
import torch
from termcolor import colored
import json
import random
import pandas as pd

# Import LLaVA related modules from utils
from utils.run_hf import run_proxy

# Import prompts
from utils.prompts import RID_prompt

# Import data utility functions
from utils.data_utils import get_item_data, DATASET_CONFIGS # DATASET_CONFIGS is imported for consistency, though not directly used in this file's logic

hf_proxy = run_proxy()
print("HF model loaded.")

# Arguments object for LLaVA proxy
args = type('Args', (), {
    "query": None,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 1024,
})()

def lookup_drive_id(image_name: str, lookup_path: str):
    with open(lookup_path, "r") as f:
        data = json.load(f)

    for item in data:
        if str(item.get("name")) == str(image_name):
            return item.get("id")

    raise ValueError(f"Image name '{image_name}' not found in {lookup_path}")


def make_drive_download_url(file_id: str):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)
c=0
def get_model_res(prompt: str, image_id: str, current_path: str):
    """Calls the HF model proxy to get a response."""
    args.image_file = image_id
    args.query = prompt
    global c, fc
    try:
        drive_id = lookup_drive_id(image_id, lookup_path=f"data/{current_path}/drive_lookup_train.json")
        image_url = make_drive_download_url(drive_id)
    except ValueError as e:
        print(colored(f"{e} Skipping count:{c} ", "red"))
        c+=1
        return ""
    

    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            _, response = hf_proxy.run_model(image_url=image_url, args=args)
            return response
        except Exception as e:
            print(colored(f"[Attempt {attempt}/{max_retries}] {e}", "red"))
            if attempt == max_retries:
                raise
    
    


def process_rid_generation(dataset_name: str, num_ssr_examples: int = 3):
    print(f"\n--- Starting RID generation for dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"

    # NEW folder structure
    train_image_base_path = f"{base_data_path}/train"
    test_image_base_path = f"{base_data_path}/test"

    # NEW csv paths
    train_csv_path = f"{base_data_path}/train.csv"
    test_csv_path = f"{base_data_path}/test_with_labels.csv"

    ssr_result_path = f"SSR/{dataset_name}_SSR.jsonl"
    rid_result_path = f"RID/{dataset_name}_RID.jsonl"

    os.makedirs(os.path.dirname(rid_result_path), exist_ok=True)

    # Load train.csv
    try:
        train_df = pd.read_csv(train_csv_path)
        print(f"Loaded {len(train_df)} train rows from {train_csv_path}.")
    except FileNotFoundError:
        print(colored(f"Error: train.csv not found at {train_csv_path}. Skipping {dataset_name}.", "red"))
        return
    except Exception as e:
        print(colored(f"Error reading {train_csv_path}: {e}", "red"))
        return

    # Load test_with_labels.csv
    try:
        test_df = pd.read_csv(test_csv_path)
        print(f"Loaded {len(test_df)} test rows from {test_csv_path}.")
    except FileNotFoundError:
        print(colored(f"Error: test_with_labels.csv not found at {test_csv_path}. Skipping {dataset_name}.", "red"))
        return
    except Exception as e:
        print(colored(f"Error reading {test_csv_path}: {e}", "red"))
        return

    # Load ssr results
    try:
        ssr_lines = [json.loads(line) for line in open(ssr_result_path, "r").readlines()]
        print(f"Loaded {len(ssr_lines)} ssr results from {ssr_result_path}.")
    except FileNotFoundError:
        print(colored(f"Error: ssr result file not found at {ssr_result_path}. Skipping {dataset_name}.", "red"))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {ssr_result_path}. Skipping {dataset_name}.", "red"))
        return

    def get_analog_rules(indices_to_use: list):
        rules = "No rules yet."
        for idx in indices_to_use:
            if 0 <= idx < len(train_df):
                row = train_df.iloc[idx]

                image_id = str(row["image_id"])
                text_content = str(row["transcriptions"]) if not pd.isna(row["transcriptions"]) else ""

                if not image_id or not text_content:
                    print(colored(f"Warning: Missing image_id or transcriptions for train item {idx}. Skipping.", "yellow"))
                    continue

                # pass image as path only (no guessing extensions)
                # image_file_path = os.path.join(train_image_base_path, image_id)

                input_prompt = RID_prompt.format(
                    org_sent=text_content,
                    rules=rules
                )

                response = get_model_res(input_prompt, image_id, dataset_name)

                print(colored(f"\n--- Input for ssr index {idx} ---", "green"))
                print(colored(input_prompt, "green"))
                print(colored(f"\n--- Model Output for ssr index {idx} ---", "blue"))
                print(colored(response, "blue"))

                if "Updated rules:" in response:
                    rules = response.split("Updated rules:", 1)[-1].strip()
                else:
                    print(colored("Warning: 'Updated rules:' tag not found. Keeping previous rules.", "yellow"))
            else:
                print(colored(f"Warning: ssr index {idx} out of bounds. Skipping.", "yellow"))
        return rules

    # Continuation logic
    start_idx = 0
    if os.path.exists(rid_result_path):
        with open(rid_result_path, "r") as f_read:
            file_lines = f_read.readlines()
            start_idx = len(file_lines)
            if start_idx > 0:
                ssr_lines = ssr_lines[start_idx:]
                print(colored(f"Continuing RID generation from index: {start_idx}", "cyan"))

    with open(rid_result_path, "a") as f_write:
        for idx_offset, item in enumerate(ssr_lines):
            current_test_index = item["index"]
            absolute_idx = idx_offset + start_idx

            if "samples" in item and len(item["samples"]) >= num_ssr_examples:
                examples = item["samples"][:num_ssr_examples]
            elif "example" in item and len(item["example"]) >= num_ssr_examples:
                examples = item["example"][:num_ssr_examples]
            else:
                print(colored(f"Warning: Not enough ssr samples for index {absolute_idx}. Skipping.", "yellow"))
                continue

            reversed_examples = examples[::-1]

            print(f"\nProcessing test item {current_test_index} (ssr list index {absolute_idx})...")
            print(f"Using ssr samples (forward): {examples}")
            print(f"Using ssr samples (backward): {reversed_examples}")

            analog_rules = get_analog_rules(examples)
            re_analog_rules = get_analog_rules(reversed_examples)

            result = {
                "index": current_test_index,
                "forward": analog_rules,
                "backward": re_analog_rules
            }

            json.dump(result, f_write)
            f_write.write("\n")
            f_write.flush()

    print(f"\n--- Finished RID generation for dataset: {dataset_name} ---")

# --- Main execution for all datasets ---
if __name__ == "__main__":
    datasets_to_process = ["misogyny/malayalam", "misogyny/tamil"]

    for dataset in datasets_to_process:
        process_rid_generation(dataset)

    print("\nAll datasets processed for RID generation.")
