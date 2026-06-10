# Requires transformers >= 5.9.0, which ships native support for the hrm_text model class:

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "sapientinc/HRM-Text-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
).cpu().eval()

# synth,cot composite — reasoning / CoT style (see Disclaimer for other modes)
condition = "<|quad_end|><|object_ref_end|>"
prompt = f"<|im_start|>{condition}Explain why the sky is blue.<|im_end|>"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# Mark the prompt as a single bidirectional prefix block — see "PrefixLM mask" below.
inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=False))

