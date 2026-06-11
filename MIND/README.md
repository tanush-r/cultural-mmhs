# MIND
The official pytorch implement of - **MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection**.

## Install

1. Clone the repo
```
git clone https://github.com/destroy-lonely/MIND.git
cd MIND
```

2. Install Package
```
conda create -n mind python=3.10 -y
conda activate mind
pip install -r requirements.txt
```

## Dataset

Please obtain FHM, HarM, and MAMI, and place them in the following directories: 
```
MIND/
├── data/
│   ├── FHM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   ├── HarM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   └── MAMI/
│       ├── images/
│       │   └── ...
│       ├── test.jsonl
│       └── train.jsonl
└── ...
```

## Quick Start

1. Similar Sample Retrieval
```
python SSR.py \
--datasets HarM FHM MAMI
```

2. Relevant Insight Derivation
```
python RID.py \
--model_path liuhaotian/llava-v1.5-13b \
--datasets HarM FHM MAMI
```

3. Insight-Augmented Inference
```
python IAI.py \
--model_path liuhaotian/llava-v1.5-13b \
--datasets HarM FHM MAMI
```
