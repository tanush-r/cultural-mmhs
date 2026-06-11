import os
import json
import torch
import random
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ----------------------------
# Seeds (same as your original)
# ----------------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

# ----------------------------
# Load HF token
# ----------------------------
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

# ----------------------------
# Same helper functions (kept)
# ----------------------------
def image_parser(args):
    if isinstance(args.image_file, list):
        return args.image_file
    return [args.image_file]


def load_image(image_file):
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    elif isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


# ----------------------------
# New: lookup drive id by name
# ----------------------------
def lookup_drive_id(image_name: str, lookup_path: str = "drive_lookup.json"):
    with open(lookup_path, "r") as f:
        data = json.load(f)

    for item in data:
        if str(item.get("name")) == str(image_name):
            return item.get("id")

    raise ValueError(f"Image name '{image_name}' not found in {lookup_path}")


def make_drive_download_url(file_id: str):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ----------------------------
# Same class name + method
# ----------------------------
class run_proxy():
    def __init__(self, model_path=None, model_base=None) -> None:
        # keeping signature same, but unused now
        self.model_name = "Qwen/Qwen3-VL-8B-Instruct:together"

    def run_model(self, args, image_url, only_encode_images=False):
        # args.query is the text prompt (same as before)
        qs = args.query

        # args.image_file will now be "79" etc (or list)
        image_files = image_parser(args)

        # take first image only (your original code effectively does one generation call)
        image_name = str(image_files[0])

        # prompt output: keep same behavior (returning prompt + output)
        prompt = qs

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": qs},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )

        outputs = completion.choices[0].message.content.strip()
        return prompt, outputs


# ----------------------------
# Example usage:
# prompt, out = run_proxy(None, None).run_model(args)
# print(out)
# ----------------------------