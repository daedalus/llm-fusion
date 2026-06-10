import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./HRM-Text-1B"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
)

# Test 1: raw text (our current approach)
text = "The capital of France is"
ids = tok.encode(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(input_ids=ids)
logit = out.logits[0, -1, :]
top5 = logit.topk(5)
print("Test 1 - Raw text:")
for i, id in enumerate(top5.indices.tolist()):
    print(f"  [{id:>5}] {tok.decode([id])!r} ({top5.values[i].item():.2f})")

# Test 2: chat format + token_type_ids
prompt = "<|im_start|>The capital of France is<|im_end|>"
ids = tok.encode(prompt, return_tensors="pt").to(model.device)
tt = torch.ones_like(ids)
with torch.no_grad():
    out = model(input_ids=ids, token_type_ids=tt)
logit = out.logits[0, -1, :]
top5 = logit.topk(5)
print("\nTest 2 - Chat format + tti:")
for i, id in enumerate(top5.indices.tolist()):
    print(f"  [{id:>5}] {tok.decode([id])!r} ({top5.values[i].item():.2f})")

# Test 3: chat format + condition tag + token_type_ids
prompt = "<|im_start|><|direct|>The capital of France is<|im_end|>"
ids = tok.encode(prompt, return_tensors="pt").to(model.device)
tt = torch.ones_like(ids)
with torch.no_grad():
    out = model(input_ids=ids, token_type_ids=tt)
logit = out.logits[0, -1, :]
top5 = logit.topk(5)
print("\nTest 3 - Chat + <|direct|> + tti:")
for i, id in enumerate(top5.indices.tolist()):
    print(f"  [{id:>5}] {tok.decode([id])!r} ({top5.values[i].item():.2f})")

# Test 4: chat format + cot condition + token_type_ids
prompt = "<|im_start|><|cot|>The capital of France is<|im_end|>"
ids = tok.encode(prompt, return_tensors="pt").to(model.device)
tt = torch.ones_like(ids)
with torch.no_grad():
    out = model(input_ids=ids, token_type_ids=tt)
logit = out.logits[0, -1, :]
top5 = logit.topk(5)
print("\nTest 4 - Chat + <|cot|> + tti:")
for i, id in enumerate(top5.indices.tolist()):
    print(f"  [{id:>5}] {tok.decode([id])!r} ({top5.values[i].item():.2f})")
