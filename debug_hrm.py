import torch
from transformers import AutoModelForCausalLM

model_path = "./HRM-Text-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
)

text = "The capital of France is"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path)

ids = tok.encode(text, return_tensors="pt").to(model.device)
print(f"Input IDs: {ids.tolist()[0]}")
print(f"Input decoded: {tok.decode(ids[0])}")

for step in range(10):
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0, -1, :]
    top_id = logits.argmax().item()
    top_str = tok.decode([top_id])
    print(f"Step {step}: top_id={top_id}, token={top_str!r}, logit={logits[top_id].item():.2f}")
    ids = torch.cat([ids, torch.tensor([[top_id]], device=model.device)], dim=-1)

print("\nFull:", tok.decode(ids[0]))
