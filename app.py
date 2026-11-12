import streamlit as st


ADAPTER_ID = "crazyintrovert7/qwen2.5-coder-7b-text-to-sql-lora"
BASE_MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"

SYSTEM_PROMPT = (
    "You translate natural-language questions into SQL. Use only the tables and "
    "columns in the provided schema. Return one executable SQL query only, "
    "without Markdown or explanation. Never execute the query."
)


@st.cache_resource(show_spinner="Loading the base model and LoRA adapter...")
def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required to load this 4-bit model. Run the app in "
            "Google Colab with a GPU runtime or on a CUDA-enabled machine."
        )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    return model, tokenizer


def clean_sql(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


st.set_page_config(page_title="Text-to-SQL", page_icon="🧠")
st.title("Text-to-SQL")
st.caption("Ask a question about your database schema and get a SQL query.")

schema = st.text_area(
    "Database schema",
    placeholder=(
        "CREATE TABLE customers (id INTEGER, name TEXT);\n"
        "CREATE TABLE orders (id INTEGER, customer_id INTEGER, total REAL);"
    ),
    height=180,
)
question = st.text_input(
    "Question",
    placeholder="Which customers have spent the most?",
)
max_new_tokens = st.slider("Maximum SQL length (tokens)", 64, 512, 256, 32)

if st.button("Generate SQL", type="primary"):
    if not schema.strip() or not question.strip():
        st.warning("Enter both a database schema and a question.")
    else:
        model, tokenizer = load_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Schema:\n{schema.strip()}\n\nQuestion:\n{question.strip()}",
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

        with st.spinner("Generating SQL..."):
            import torch

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        sql = clean_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))
        st.subheader("Generated SQL")
        st.code(sql or "-- The model did not return a SQL query.", language="sql")
        st.caption("Review the query before running it against a database.")
