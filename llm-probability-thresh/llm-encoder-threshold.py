from utils import login_to_huggingface, choose_device, jsonl_to_list, get_custom_word_ids
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json

# MODEL
hf_token = login_to_huggingface(Path(".env"))
device = choose_device("auto")

model_id = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

model.to(device)
model.eval()

# DATA

data = jsonl_to_list("ged-syntax-probing-main/datasets/fce/json/fce.dev.json")

with open("llm-probability-thresh/fce-dev-paragraph-labels.json") as f:
    label_data = json.load(f)

# LOOP

big_bad_data = []

for essay_no, (essay, label_essay) in enumerate(zip(data, label_data)):

    big_data = []

    for sentence_no, (sentence, label_paragraph) in enumerate(zip(essay, label_essay)):
        print(f"essay {essay_no + 1}/{len(data)} | sentence {sentence_no + 1}/{len(essay)}")
        tsv_spans = label_paragraph["spans"]

        encoding = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True).to(device)
        offset_mapping = encoding["offset_mapping"][0]

        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        special_ids = set(tokenizer.all_special_ids)
        word_ids = get_custom_word_ids(input_ids, offset_mapping, tsv_spans, special_ids)

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
                "word_id": word_ids[idx],
                "probability": target_prob,
                "rank": rank,
                "percentile": percentile
            })
    
        big_data.append(results)
    
    big_bad_data.append(big_data)

with open("llm-probability-thresh/data.json", "w") as f:
    json.dump(big_bad_data, f, indent=4)