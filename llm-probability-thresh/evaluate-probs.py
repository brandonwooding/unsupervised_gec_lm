import json
from statistics import mean


with open("llm-probability-thresh/fce-dev-paragraph-labels.json", "r") as file:
    label_data = json.load(file)

with open("llm-probability-thresh/data.json", "r") as file:
    prob_data = json.load(file)


prob_data_flat = [paragraph for essay in prob_data for paragraph in essay]
label_data_flat = [paragraph for essay in label_data for paragraph in essay]

assert len(prob_data_flat) == len(label_data_flat)

skipped_paragraphs = 0
skipped_label_tokens = 0

c_probs = []
c_percs = []
c_ranks = []

i_probs = []
i_percs = []
i_ranks = []


for paragraph_idx, prob_paragraph in enumerate(prob_data_flat):
    labels = label_data_flat[paragraph_idx]["labels"]

    word_ids = [
        token["word_id"]
        for token in prob_paragraph
        if token["word_id"] is not None
    ]

    if not word_ids:
        skipped_paragraphs += 1
        skipped_label_tokens += len(labels)
        continue

    model_word_count = max(word_ids) + 1
    label_word_count = len(labels)

    if model_word_count != label_word_count:
        print(label_data_flat[paragraph_idx]["tokens"])
        skipped_paragraphs += 1
        skipped_label_tokens += len(labels)
        continue

    word_probs = [1.0] * label_word_count
    word_percs = [1.0] * label_word_count
    word_ranks = [1] * label_word_count

    for token in prob_paragraph:
        word_id = token["word_id"]
        if word_id is None:
            continue

        word_probs[word_id] = min(word_probs[word_id], token["probability"])
        word_percs[word_id] = min(word_percs[word_id], token["percentile"])
        word_ranks[word_id] = max(word_ranks[word_id], token["rank"])

    for label, prob, perc, rank in zip(labels, word_probs, word_percs, word_ranks):
        if label == "c":
            c_probs.append(prob)
            c_percs.append(perc)
            c_ranks.append(rank)
        elif label == "i":
            i_probs.append(prob)
            i_percs.append(perc)
            i_ranks.append(rank)
        else:
            raise ValueError(f"unexpected label: {label}")


def safe_mean(values):
    return mean(values) if values else float("nan")


print(f"paragraphs: {len(prob_data_flat)}")
print(f"skipped_paragraphs: {skipped_paragraphs}")
print(f"used_paragraphs: {len(prob_data_flat) - skipped_paragraphs}")
print(f"skipped_label_tokens: {skipped_label_tokens}")
print()

print("correct labels")
print(f"count: {len(c_probs)}")
print(f"avg_probability: {safe_mean(c_probs)}")
print(f"avg_percentile: {safe_mean(c_percs)}")
print(f"avg_rank: {safe_mean(c_ranks)}")
print()

print("incorrect labels")
print(f"count: {len(i_probs)}")
print(f"avg_probability: {safe_mean(i_probs)}")
print(f"avg_percentile: {safe_mean(i_percs)}")
print(f"avg_rank: {safe_mean(i_ranks)}")
