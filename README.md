# Text-to-SQL Fine-Tuned Model (QLoRA)

A small Streamlit app for trying your fine-tuned Qwen2.5-Coder-7B LoRA adapter. Enter a database schema and a natural-language question to generate a SQL query.

## Model and dataset

- **Base model:** `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
- **LoRA adapter:** [crazyintrovert7/qwen2.5-coder-7b-text-to-sql-lora](https://huggingface.co/crazyintrovert7/qwen2.5-coder-7b-text-to-sql-lora)
- **Dataset:** `b-mc2/sql-create-context` (3,000 examples, as reported for this fine-tune)
- **Method:** QLoRA

The adapter repository provides its tokenizer and adapter weights. On first inference, the app downloads those files and the base model from Hugging Face and caches them locally.

## Training results

The reported Colab T4 run completed in under 5 minutes. Training loss reportedly decreased from 2.30 to 0.58 over 60 steps. These are training-run figures, not independent evaluation results; test generated SQL against representative questions and schemas before relying on it.

## Requirements

- Python 3.10 or newer
- A CUDA-capable NVIDIA GPU with enough VRAM for the 4-bit 7B model and inference
- Internet access for the initial model download

The app intentionally requires CUDA rather than trying to load a 7B model into CPU memory. Google Colab with a GPU runtime is a convenient way to run it.

## Run locally

From this project directory:

```bash
python -m venv .venv
```

Activate the virtual environment, then install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, paste in your schema, enter a question, and select **Generate SQL**. The first generation loads and caches the model; later generations reuse it.

## Run in Google Colab

1. Select **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
2. Upload or clone this project into the Colab runtime.
3. Install dependencies with `!pip install -r requirements.txt`.
4. Start Streamlit using a Colab-compatible tunnel or access method, then open the URL it provides.

The model is public, so Hugging Face authentication is not normally required. The adapter files currently occupy about 173 MB; the base model download is several gigabytes. Allow additional disk space for the Hugging Face cache.

## Notes

- The app uses the Qwen tokenizer's chat template and asks the model for one SQL query without Markdown or explanation.
- Generated SQL is **not executed**. Review it for correctness and safety before running it against any database.
- Model output quality depends on the schema, question, and prompt format used during fine-tuning. If the fine-tuning notebook used a custom prompt template, update the prompt construction in `app.py` to match it.
- The reported training time and loss are training-run measurements, not independently benchmarked results.
