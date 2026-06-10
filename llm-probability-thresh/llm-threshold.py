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

test_sentence = "The man enjoy playing cricket and hate intersectionality."
test_mask = f"The man {tokenizer.mask_token} playing cricket."
target = "enjoy"
print(test_sentence)

encoding = tokenizer(test_sentence, return_tensors="pt").to(device)

input_ids = encoding["input_ids"][0]
attention_mask = encoding["attention_mask"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(tokens)

special_ids = set(tokenizer.all_special_ids)

results = []

for idx, original_token_id in enumerate(input_ids):
    original_token_id = original_token_id.item()

    if original_token_id in special_ids:
        continue

    masked_input_ids = input_ids.clone()
    masked_input_ids[idx] = tokenizer.mask_token_id

    inputs = {
        "input_ids": masked_input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0)
    }

    with torch.no_grad():
        logits = model(**inputs).logits[0, idx]

    probs = torch.softmax(logits, dim=-1)

    target_prob = probs[original_token_id].item()

    rank = (probs > probs[original_token_id]).sum().item() + 1

    percentile = (probs < probs[original_token_id]).float().mean().item()

    results.append({
        "position": idx,
        "token": tokens[idx],
        "token_id": original_token_id,
        "probability": target_prob,
        "rank": rank,
        "percentile": percentile
    })


word_ids = encoding.word_ids(batch_index=0)
print(word_ids)

print(results)
    
"""inputs = tokenizer(test_mask, return_tensors="pt").to(device)

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
"""