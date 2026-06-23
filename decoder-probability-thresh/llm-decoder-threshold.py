from utils import login_to_huggingface, choose_device, jsonl_to_list, get_custom_word_ids
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# MODEL
hf_token = login_to_huggingface(Path(".env"))
device = choose_device("auto")

model_id = "google/gemma-4-12B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
)

model.to(device)
model.eval()

# DATA

data = jsonl_to_list("ged-syntax-probing-main/datasets/fce/json/fce.dev.json")

with open("llm-probability-thresh/fce-dev-paragraph-labels.json") as f:
    label_data = json.load(f)

# Process only the first MAX_ESSAYS essays. Set this to a small number (e.g.
# 2) for a quick smoke test, or to None to run the whole dataset.
MAX_ESSAYS = None
if MAX_ESSAYS is not None:
    data = data[:MAX_ESSAYS]
    label_data = label_data[:MAX_ESSAYS]

special_ids = set(tokenizer.all_special_ids)

# A causal model predicts a token only from what comes BEFORE it, so the very
# first token of a paragraph has nothing to predict it from. We give it some
# left context by sticking a beginning-of-sequence token on the front. It is
# a special token like any other, so it is never itself scored, but the
# model can use it to make a prediction for the first real word.
bos_id = tokenizer.bos_token_id
if bos_id is None:
    bos_id = tokenizer.eos_token_id

# LOOP

big_bad_data = []

for essay_no, (essay, label_essay) in enumerate(zip(data, label_data)):

    big_data = []

    for sentence_no, (sentence, label_paragraph) in enumerate(zip(essay, label_essay)):
        print(f"essay {essay_no + 1}/{len(data)} | paragraph {sentence_no + 1}/{len(essay)}")
        tsv_spans = label_paragraph["spans"]

        encoding = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        offset_mapping = encoding["offset_mapping"][0]

        needs_bos = len(input_ids) == 0 or input_ids[0].item() not in special_ids
        if bos_id is not None and needs_bos:
            input_ids = torch.cat([torch.tensor([bos_id]), input_ids])
            attention_mask = torch.cat([torch.tensor([1]), attention_mask])
            offset_mapping = torch.cat([torch.tensor([[0, 0]]), offset_mapping])

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        word_ids = get_custom_word_ids(input_ids, offset_mapping, tsv_spans, special_ids)

        inputs = {
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
        }

        # One forward pass over the whole paragraph gives a next-token
        # prediction for every position at once, because causal attention
        # already only lets position idx see positions 0..idx. So unlike the
        # encoder version, we don't need to re-run the model once per token.
        with torch.no_grad():
            logits = model(**inputs).logits[0]

        results = []

        for idx in range(1, len(input_ids)):
            original_token_id = input_ids[idx].item()

            if original_token_id in special_ids:
                continue

            # logits[idx - 1] is the model's predicted distribution for the
            # token at position idx, based only on tokens before it.
            probs = torch.softmax(logits[idx - 1], dim=-1)

            target_prob = probs[original_token_id].item()

            rank = (probs > probs[original_token_id]).sum().item() + 1

            percentile = (probs < probs[original_token_id]).float().mean().item()

            results.append({
                "position": idx,
                "token": tokens[idx],
                "token_id": original_token_id,
                "word_id": word_ids[idx],
                "probability": target_prob,
                "rank": rank,
                "percentile": percentile
            })

        big_data.append(results)

    big_bad_data.append(big_data)

with open("decoder-probability-thresh/data.json", "w") as f:
    json.dump(big_bad_data, f, indent=4)
