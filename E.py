# IMPORTANT: Please use transformers<4.56.0 to avoid compatibility issues. We recommend transformers==4.54.1 or earlier versions.

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./Ouro-1.4B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

