import os
import json
import random
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ----------------------------
# Load HF token
# ----------------------------
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

image_url="https://drive.google.com/uc?export=download&id=1W9OvSWj4hqSaMyzihQnADw59Zu4Vxw-c"
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

completion = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct:together",
    # model="openai/gpt-oss-20b:hyperbolic",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is image"},
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
print(outputs)