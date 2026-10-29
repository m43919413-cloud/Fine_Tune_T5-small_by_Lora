# T5-LoRA Summarizer

Fine-tuning **t5-small** with **LoRA (PEFT)** for dialogue summarization on the
SAMSUM dataset.

A lightweight, parameter-efficient summarization model — the LoRA adapter is
only ~3 MB, while the base model is loaded separately at inference time.

---

## Overview

| Item | Value |
|---|---|
| **Base model** | t5-small (60M params) |
| **Fine-tuning method** | LoRA (r=8, alpha=32, target: q, v) |
| **Dataset** | knkarthick/samsum (dialogue → summary) |
| **Trainable params** | ~0.6% of total |
| **Task** | Abstractive dialogue summarization |
| **Language** | English |

The model was fine-tuned using Seq2SeqTrainer for 3 epochs with a learning
rate of 3e-4, then the LoRA adapter was saved and merged with the base model
at inference time.

---

## Project structure

- Fine_Tune_LoRA_T5.py — training script (run on Colab / GPU)
- Inference_T5.py — inference / usage script
- adapter_config.json — LoRA adapter config
- adapter_model.safetensors — LoRA adapter weights (~3 MB)
- tokenizer.json — tokenizer files
- requirements.txt
- .gitignore
- README.md

---

## Installation

Clone the repo and set up a virtual environment:

    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO

    python -m venv venv
    source venv/bin/activate          # Linux / macOS
    # venv\Scripts\activate           # Windows

    pip install -r requirements.txt

### requirements.txt

    transformers>=4.40.0
    peft>=0.10.0
    torch>=2.0.0
    datasets>=2.18.0
    accelerate>=0.28.0

### .gitignore

    venv/
    __pycache__/
    *.pyc
    .ipynb_checkpoints/
    *.bin
    *.pt
    *.pth
    *.ckpt
    .DS_Store

Note: the *.bin rule prevents the base model weights (pytorch_model.bin, ~240 MB)
from being pushed by mistake. adapter_model.safetensors uses the .safetensors
extension, so it IS committed — that is intentional.

---

## Usage

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    import torch

    ADAPTER_PATH = "./"               # folder containing the adapter files
    BASE_MODEL   = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()

    def summarize(dialogue: str) -> str:
        prompt = "summarize: " + dialogue          # required — same as training
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512)
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            early_stopping=True,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    dialogue = """Amanda: I baked cookies. Do you want some?
    Jerry: Sure!
    Amanda: I'll bring you tomorrow :-)"""

    print(summarize(dialogue))
    # → Amanda baked cookies. Jerry will bring Amanda tomorrow.

Important: the "summarize: " prefix is mandatory — it was used during training
and must be included at inference, otherwise the output degrades badly.

---

## Examples

| Input (dialogue) | Model output | Quality |
|---|---|---|
| SAMSUM-style conversation | Amanda baked cookies. Jerry will bring Amanda tomorrow. | Good structure, minor role swap |
| English news article | repetitive / off-topic | Out-of-domain — model trained on dialogues only |
| Persian text | — | Not supported (t5-small is English-only) |

---

## Limitations

- Role / entity swapping. t5-small has limited capacity for tracking who does
  what in multi-turn dialogues. Occasionally the subject and object are inverted
  (e.g. "Jerry will bring Amanda" instead of the reverse).
- In-domain only. Trained on casual English dialogues (SAMSUM). News, technical
  text, and formal documents produce low-quality summaries.
- English only. t5-small does not support Persian or other languages.
- Short outputs. Maximum summary length during training was 128 tokens, so the
  model tends to produce 1–2 sentence summaries.

---

## Possible improvements

To increase summary quality and role-tracking accuracy:

1. Larger base model — swap t5-small → t5-base or google/flan-t5-base (biggest
   impact; t5-small is simply too small for entity tracking).
2. Higher LoRA rank — r=8 → r=16.
3. More LoRA target modules — ["q", "v"] → ["q", "k", "v", "o", "wi", "wo"].
4. More epochs / lower LR — 3 → 5 epochs, 3e-4 → 1e-4.
5. Stronger beam search — num_beams=6.

For Persian summarization, use google/mt5-small or mt5-base instead, and
fine-tune on a Persian summarization dataset.

---

## Training details

    LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )

    Seq2SeqTrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=3e-4,
        predict_with_generate=True,
        eval_strategy="epoch",
    )

Inputs were prefixed with "summarize: " and truncated to 512 tokens; summaries
were truncated to 128 tokens.

---

## Acknowledgements

- t5-small — Google Research
- SAMSUM — Samsung R&D
- PEFT — Hugging Face