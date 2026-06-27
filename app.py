import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import base64
import io, os
from dotenv import load_dotenv
from prompts import cultural_prompt_drishtikon, cultural_agent

load_dotenv(override=True)

client = InferenceClient(
    model="Qwen/Qwen3-VL-8B-Instruct:novita",
    api_key=os.environ["HF_TOKEN"]
)

if "rules" not in st.session_state:
    st.session_state.rules = ""
if "output" not in st.session_state:
    st.session_state.output = ""

def encode_image(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def run_agent(image):
    img_b64 = encode_image(image)
    SYSTEM_PROMPT = cultural_agent.format(org_sent="", rules=st.session_state.rules)
    print(SYSTEM_PROMPT)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        }
    ]

    resp = client.chat_completion(
        messages=messages,
        max_tokens=500
    )

    return resp.choices[0].message.content

st.set_page_config(page_title="Multimodal Hate Speech Detection", layout="wide")

st.title("Multimodal Hate Speech Detection")

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    image = None
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    run_btn = st.button("Analyze")

with right:
    st.subheader("Cultural Agent Reasoning")

    output_box = st.empty()

    if run_btn:
        if image is None:
            output_box.warning("Please upload an image first.")
        else:
            with st.spinner("Running cultural reasoning model..."):
                output = run_agent(image)

                st.session_state.rules = output.split("Updated rules:")[1]
                st.session_state.output = output

        output_box.markdown(
            f"""
            <div style="
                background-color:#111827;
                padding:16px;
                border-radius:10px;
                color:#E5E7EB;
                font-family: monospace;
                white-space: pre-wrap;
            ">
            {st.session_state.output }
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.subheader("Existing rules:")
    st.text(st.session_state.rules)