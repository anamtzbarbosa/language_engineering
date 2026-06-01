# Language Engineering (KTH - EECS)

This repository focus on empirical evaluations of Transformer architectures, language model behavior, and layer-by-layer probing analysis.

---

## Repository Structure & Core Files

The project is structured around specific Jupyter Notebooks for training, alignment, and layer-by-layer probing experiments:

### 1. Training & Core Models
* **`Next_word.ipynb`**: Notebook for training and evaluating the autoregressive Next-Word Prediction model.
* **`Fill_in_the_blanks.ipynb`**: Notebook for training and evaluating the Masked Language Model (Unmasking).
* **`self_attention.py`**: Custom implementation of the Multi-Head Attention mechanisms and base Transformer architecture layers.
* **`tokenizer.py` & `tokenizer.json`**: Tokenization pipeline, subtoken management, and alignment logic.

### 2. Alignment & Dataset Processing
* **`Aligment_tensor.ipynb`**: Alignment of tokens and subtokens to map model hidden states with linguistic features.
* **`multipos_dict.json` & `word_to_pos_dict.json`**: Pre-processed dictionaries utilizing an optimized SpaCy pipeline and regex filtering to manage Part-of-Speech (POS) tags.

### 3. Diagnostic Probing Experiments
* **`Probing_experiment_next_word.ipynb`**: Layer-by-layer diagnostic probe training and high-uncertainty case analysis for the Next-Word model.
* **`Probing_experiment_fill_in_the_blanks.ipynb`**: Layer-by-layer diagnostic probe training and high-uncertainty case analysis for the Fill-in-the-Blanks model.
* **`Plots.ipynb`**: Script and logic dedicated to generating evaluation charts, accuracy metrics, and layer transition plots.
* **`Settings.py`**: Global experimental settings, hyperparameters, and path definitions.

### 4. Checkpoints & Pre-trained Probes
* **`last_checkpoint_next_word.pt`**: Saved weights for the trained Next-Word architecture.
* **`last_checkpoint_unmasking.pt`**: Saved weights for the trained Fill-in-the-Blanks architecture.
* **`tags_probes_5k_first_subtoken.pt`**: Saved diagnostic probe weights mapped to the first subtoken for POS evaluation.

---

## Project Workflow

To replicate the empirical study and probing analysis, the files should be evaluated in the following sequence:

1. **Model Training**: Run `Next_word.ipynb` and `Fill_in_the_blanks.ipynb` to train the base architectures on the TinyStories dataset or load the provided `.pt` checkpoints.
2. **Subtoken Alignment**: Execute `Aligment_tensor.ipynb` to process the text inputs, align subtoken splits with their corresponding linguistic labels, and map the tensors.
3. **Probing Experiments**: Run `Probing_experiment_next_word.ipynb` and `Probing_experiment_fill_in_the_blanks.ipynb` to train the linear diagnostic probes across all layers (Layer 0 to Layer 3) and extract ambiguous tokens.
4. **Visualization**: Use `Plots.ipynb` to render the final comparative curves and uncertainty behavior plots across different network layers.

---
