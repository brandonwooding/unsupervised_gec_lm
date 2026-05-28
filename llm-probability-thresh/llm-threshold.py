from utils import login_to_huggingface, choose_device
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

hf_token = login_to_huggingface(Path(".env"))
device = choose_device("auto")

model_id = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

model.to(device)
model.eval()

test_sentence = "The man enjoys playing cricket."
test_mask = f"The man {tokenizer.mask_token} playing cricket."
target = "enjoy"

    
inputs = tokenizer(test_mask, return_tensors="pt").to(device)

mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
if len(mask_positions) != 1:
    raise ValueError(
        f"Expected exactly one mask token {tokenizer.mask_token!r}, "
        f"found {len(mask_positions)} in: {test_mask!r}"
    )
mask_pos = mask_positions.item()

with torch.no_grad():
    logits = model(**inputs).logits[0, mask_pos]

probs = torch.softmax(logits, dim=-1)

target_ids = tokenizer.encode(target, add_special_tokens=False)
print(target_ids)
print(tokenizer.convert_ids_to_tokens(target_ids))
print(tokenizer.decode(target_ids))
target_id = target_ids[0]

target_prob = probs[target_id].item()
print(target_prob)

percentile = (probs < target_prob).float().mean().item()
print(percentile)

rank = (probs > target_prob).sum().item() + 1
print(rank)
