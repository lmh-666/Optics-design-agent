import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/root/.cache/modelscope/hub/models/ckdckd/OpticsGPT-v0"

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True
)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True
)

prompt = "什么是几何光学？"
messages = [{"role": "user", "content": prompt}]

try:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
except Exception:
    text = prompt

inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("开始生成...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# 只取新生成的 token
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("------ Output ------")
print(response)