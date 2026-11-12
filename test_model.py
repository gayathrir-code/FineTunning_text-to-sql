from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1. Point to your HF repo
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_MODEL = "crazyintrovert7/qwen2.5-coder-7b-text-to-sql-lora" # CHANGE THIS

print("Loading base model... (this might take a few minutes on CPU)")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32) # CPU needs float32

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

# 2. Format the prompt
prompt = """<|im_start|>system
You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and a context. You must output the SQL query that answers the question.
<|im_end|>
<|im_start|>user
### Context:
CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT, hire_date DATE)

### Question:
What are the names of the employees in the 'Engineering' department?
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([prompt], return_tensors="pt")

print("\nGenerating SQL...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)

response = tokenizer.batch_decode(outputs)[0]
sql = response.split("<|im_start|>assistant")[1].replace("<|im_end|>", "").strip()

print("\n--- GENERATED SQL ---")
print(sql)