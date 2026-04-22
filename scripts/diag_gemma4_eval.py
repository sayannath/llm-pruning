"""Diagnostic: verify _tokenize_prompt fix for Gemma4 special-token handling."""
import torch
import torch.nn.functional as F
import os, sys
sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM, AutoTokenizer

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
hf_id = "google/gemma-4-E4B-it"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)

raw_prompt = "Question: What is the capital of France?\nA. Berlin\nB. Madrid\nC. Paris\nD. Rome\nAnswer:"

# --- Correct approach: apply_chat_template with tokenize=True ---
prompt_ids = tok.apply_chat_template(
    [{"role": "user", "content": raw_prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
if not isinstance(prompt_ids, torch.Tensor):
    prompt_ids = prompt_ids["input_ids"]

print(f"prompt_ids.shape (correct): {prompt_ids.shape}  prompt_len={prompt_ids.shape[1]}")

# --- Old broken approach for comparison ---
formatted_str = tok.apply_chat_template(
    [{"role": "user", "content": raw_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
broken_ids = tok(formatted_str, return_tensors="pt", add_special_tokens=False)["input_ids"]
print(f"prompt_ids.shape (broken):  {broken_ids.shape}  prompt_len={broken_ids.shape[1]}")
print()

for c in ["A", "B", "C", "D"]:
    ids = tok(" " + c, return_tensors="pt", add_special_tokens=False)["input_ids"]
    print(f"  ' {c}' -> ids={ids[0].tolist()}, len={ids.shape[1]}")
print()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    hf_id, token=token, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
device = next(model.parameters()).device

print("\n=== Log-prob scores (correct tokenisation) ===")
with torch.no_grad():
    for c in ["A", "B", "C", "D"]:
        choice_ids = tok(" " + c, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([prompt_ids, choice_ids], dim=1).to(device)
        pl = prompt_ids.shape[1]
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1)
        tlp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(pl - 1, 0)
        per_tok = tlp[0, start:].tolist()
        score = sum(per_tok) / max(len(per_tok), 1)
        print(f"  '{c}': score={score:.3f}, per_token={[round(x,2) for x in per_tok]}")

print("\n=== Generation test ===")
with torch.no_grad():
    out = model.generate(
        prompt_ids.to(device),
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
generated = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
print(f"Generated: {repr(generated)}")
