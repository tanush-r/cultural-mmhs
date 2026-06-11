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
from utils.prompts import IAI_debater_prompt, IAI_judge_prompt
from sklearn.metrics import f1_score # Import f1_score

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
        drive_id = lookup_drive_id(image_id, lookup_path=f"data/{current_path}/drive_lookup.json")
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

def find_answer_position(answer: str):
    """Determines if the answer indicates 'harmful' or 'harmless'."""
    answer = answer.lower().strip()
    if "harmful" in answer:
        return 1
    elif "harmless" in answer:
        return 0
    return None

def process_iai_evaluation(dataset_name: str):
    """
    Performs Iterative Analog Inference (IAI) evaluation for a given dataset.
    It uses rules derived from similar memes to assess target memes.
    """
    print(f"\n--- Starting IAI evaluation for dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"

    # NEW folder structure
    train_image_base_path = f"{base_data_path}/train"
    test_image_base_path = f"{base_data_path}/test"

    # NEW csv paths
    train_csv_path = f"{base_data_path}/train.csv"
    test_csv_path = f"{base_data_path}/test_with_labels.csv"

    rid_result_path = f"RID/{dataset_name}_RID.jsonl"
    iai_result_path = f"IAI/{dataset_name}_IAI.jsonl"

    os.makedirs(os.path.dirname(iai_result_path), exist_ok=True)

    # Load train.csv (not used directly but keeping consistent with structure)
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

    try:
        rid_lines = [json.loads(line) for line in open(rid_result_path, 'r').readlines()]
        print(f"Loaded {len(rid_lines)} RID results from {rid_result_path}.")
    except FileNotFoundError:
        print(colored(f"Error: RID result file not found at {rid_result_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {rid_result_path}. Skipping {dataset_name}.", 'red'))
        return

    ratio = [0, 0]  # [total_processed, correct_predictions]
    start_idx = 0

    all_actual_labels = []
    all_predicted_labels = []

    # Continuation logic
    if os.path.exists(iai_result_path):
        with open(iai_result_path, 'r') as f_read:
            file_lines = f_read.readlines()
            start_idx = len(file_lines)

            if start_idx > 0:
                try:
                    last_result = json.loads(file_lines[-1])
                    if 'ratio' in last_result and isinstance(last_result['ratio'], list) and len(last_result['ratio']) == 2:
                        ratio = last_result['ratio']
                        if 'accuracy' in last_result:
                            print(colored(f"Dataset {dataset_name} seems to be fully processed. Delete {iai_result_path} to re-run.", 'green'))
                            return
                    else:
                        print(colored("Warning: 'ratio' key not found or malformed in last result, resetting ratio.", 'yellow'))
                        ratio = [0, 0]
                except json.JSONDecodeError:
                    print(colored("Warning: Could not decode last line for ratio, resetting ratio.", 'yellow'))
                    ratio = [0, 0]

                for i in range(start_idx):
                    try:
                        prev_item = json.loads(file_lines[i])
                        if 'actual' in prev_item and 'predict' in prev_item:
                            all_actual_labels.append(prev_item['actual'])
                            all_predicted_labels.append(prev_item['predict'] if prev_item['predict'] is not None else 1 - prev_item['actual'])
                    except json.JSONDecodeError:
                        continue

                test_df = test_df.iloc[start_idx:].reset_index(drop=True)
                rid_lines = rid_lines[start_idx:]

                print(colored(f'Continuing IAI evaluation from index: {start_idx}', 'cyan'))
                print(f"Current ratio: {ratio}")

    with open(iai_result_path, 'a') as f_write:
        for idx_offset, (row, rid_line) in enumerate(zip(test_df.itertuples(index=False), rid_lines)):
            current_absolute_idx = start_idx + idx_offset

            image_id = getattr(row, "image_id", None) or getattr(row, "fimage_id", None)

            if image_id is None:
                raise ValueError("No image_id or fimage_id found in row")

            image_id = str(image_id)
            text_content = str(getattr(row, "transcriptions")) if not pd.isna(getattr(row, "transcriptions")) else ""
            label_str = str(getattr(row, "labels")).strip().lower()

            if label_str in ["misogyny"]:
                label = 1
            elif label_str in ["not misogyny", "not-misogyny"]:
                label = 0
            else:
                raise ValueError(f"Unknown label value: {label_str}")

            if not image_id:
                print(colored(f"Warning: Missing image_id for test item {current_absolute_idx}. Skipping.", 'yellow'))
                continue

            forward_rules = rid_line['forward']
            backward_rules = rid_line['backward']

            is_equal = False

            input_debater1 = IAI_debater_prompt.format(text_content, forward_rules)
            output_debater1 = get_model_res(input_debater1, image_id, dataset_name)
            print(colored("\n--- Debater 1 Output ---", 'green'))
            print(colored(output_debater1, 'green'))
            predict_1 = output_debater1.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_1 = output_debater1.split("Thought: ")[-1]

            input_debater2 = IAI_debater_prompt.format(text_content, backward_rules)
            output_debater2 = get_model_res(input_debater2, image_id, dataset_name)
            print(colored("\n--- Debater 2 Output ---", 'yellow'))
            print(colored(output_debater2, 'yellow'))
            predict_2 = output_debater2.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_2 = output_debater2.split("Thought: ")[-1]

            if predict_1 != predict_2:
                input_judge = IAI_judge_prompt.format(text_content, predict_1, thought_1[:1200], predict_2, thought_2[:1200])
                output_judge = get_model_res(input_judge, image_id, dataset_name)
                print(colored("\n--- Judge Input ---", 'red'))
                print(colored(input_judge, 'red'))
                print(colored("\n--- Judge Output ---", 'blue'))
                print(colored(output_judge, 'blue'))
                predict = output_judge.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            else:
                is_equal = True
                predict = predict_1

            final_predict_val = find_answer_position(predict)

            if final_predict_val is None:
                final_predict_val = 1 - label

            if final_predict_val == label:
                ratio[1] += 1
            ratio[0] += 1

            all_actual_labels.append(label)
            all_predicted_labels.append(final_predict_val)

            result = {
                'index': current_absolute_idx,
                'ratio': list(ratio),
                'actual': label,
                'predict': final_predict_val,
                'text': text_content,
                'output': output_debater1 if is_equal else output_judge
            }

            result['debater1_predict'] = find_answer_position(predict_1)
            result['debater2_predict'] = find_answer_position(predict_2)

            print(f"Actual: {label}, Predict: {final_predict_val}, ratio: {ratio}")

            json.dump(result, f_write)
            f_write.write('\n')
            f_write.flush()

        accuracy = ratio[1] / ratio[0] if ratio[0] > 0 else 0

        macro_f1 = 0
        if len(all_actual_labels) > 0:
            macro_f1 = f1_score(all_actual_labels, all_predicted_labels, average='macro')

        final_summary = {'ratio': ratio, 'accuracy': accuracy, 'macro_f1': macro_f1}
        json.dump(final_summary, f_write)
        f_write.write('\n')

    print(f"\n--- Finished IAI evaluation for dataset: {dataset_name} ---")
    print(f"Final Accuracy for {dataset_name}: {accuracy:.4f} ({ratio[1]}/{ratio[0]})")
    print(f"Final Macro F1 Score for {dataset_name}: {macro_f1:.4f}")


if __name__ == "__main__":
    datasets_to_process = ["misogyny/malayalam", "misogyny/tamil"]

    for dataset in datasets_to_process:
        process_iai_evaluation(dataset)

    print("\nAll datasets processed for IAI evaluation.")