# Language Engineering (KTH - EECS)

This repository focus on empirical evaluations of Transformer architectures, language model behavior, and layer-by-layer probing analysis.

## Repository Structure

The repository is organized into two main tracks:

```text
├── 1_NextWord_vs_Unmasking/     # Probing the Linguistic Knowledge of Transformers
│   ├── data/                    # Dataset utilities (TinyStories)
│   ├── models/                  # Custom Multi-Head Attention & Transformer definitions
│   ├── probing/                 # POS tagging probes and layer extraction logic
│   └── main.py                  # Training and evaluation loops (130k iterations)
│
└── README.md
