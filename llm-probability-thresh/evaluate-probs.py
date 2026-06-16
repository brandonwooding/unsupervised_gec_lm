import json
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


with open("llm-probability-thresh/fce-dev-paragraph-labels.json", "r") as file:
    label_data = json.load(file)

with open("llm-probability-thresh/data_1.json", "r") as file:
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

def get_metrics_from_probs(c_probs, i_probs, threshold):
    correct_labels = [0]*len(c_probs) + [1]*len(i_probs)
    predicted_labels = [int(p < threshold) for p in c_probs] + [int(p < threshold) for p in i_probs]
    report = classification_report(correct_labels, predicted_labels)
    return report

print(get_metrics_from_probs(c_probs, i_probs, threshold=0.4))

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

## PLOTS

"""c_ppls = -np.log(np.array(c_probs) + 1e-9)
i_ppls = -np.log(np.array(i_probs) + 1e-9)

def prob_bins():
    return np.concatenate([np.linspace(0, 0.1, 30), np.linspace(0.1, 1.0, 21)[1:]])

def plot_overlay_hist(ax, c_vals, i_vals, title, xlabel, log_x=False, bins=None):
    c_arr, i_arr = np.array(c_vals), np.array(i_vals)
    if bins is None:
        if log_x:
            bins = np.logspace(np.log10(max(min(c_arr.min(), i_arr.min()), 1)), np.log10(max(c_arr.max(), i_arr.max())), 60)
        else:
            bins = prob_bins()
    ax.hist(c_arr, bins=bins, density=True, alpha=0.5, color='steelblue', label='correct')
    ax.hist(i_arr, bins=bins, density=True, alpha=0.5, color='tomato', label='incorrect')
    if log_x:
        ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()

def plot_cdf(ax, c_vals, i_vals, title, xlabel):
    for vals, label, color in [(c_vals, 'correct', 'steelblue'), (i_vals, 'incorrect', 'tomato')]:
        s = np.sort(vals)
        ax.plot(s, np.linspace(0, 1, len(s)), label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Density')
    ax.legend()

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
plot_overlay_hist(axes1[0], c_probs, i_probs, 'Probability', 'Min Token Probability')
plot_overlay_hist(axes1[1], c_ranks, i_ranks, 'Rank (log scale)', 'Max Token Rank', log_x=True)
fig1.suptitle('Correct vs Incorrect — Density (overlaid)', fontsize=13)
fig1.tight_layout()
fig1.savefig('llm-probability-thresh/distributions_overlay.png', dpi=150)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
plot_cdf(axes2[0], c_probs, i_probs, 'Probability CDF', 'Min Token Probability')
plot_cdf(axes2[1], c_ranks, i_ranks, 'Rank CDF', 'Max Token Rank')
fig2.suptitle('Correct vs Incorrect — CDF', fontsize=13)
fig2.tight_layout()
fig2.savefig('llm-probability-thresh/distributions_cdf.png', dpi=150)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
plot_overlay_hist(axes3[0], c_ppls, i_ppls, 'Surprisal (log density)', 'Surprisal / Token PPL (-log P)', bins=60)
plot_overlay_hist(axes3[1], c_ranks, i_ranks, 'Rank (log-log)', 'Max Token Rank', log_x=True)
axes3[0].set_yscale('log')
axes3[1].set_yscale('log')
fig3.suptitle('Correct vs Incorrect — Density (log y-axis)', fontsize=13)
fig3.tight_layout()
fig3.savefig('llm-probability-thresh/distributions_log.png', dpi=150)

rng = np.random.default_rng(42)
n_sample = min(2000, len(c_ppls), len(i_ppls))
c_idx = rng.choice(len(c_ppls), n_sample, replace=False)
i_idx = rng.choice(len(i_ppls), n_sample, replace=False)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(np.array(c_probs)[c_idx], np.log(np.array(c_ranks)[c_idx]), alpha=0.2, s=8, color='steelblue', label='correct')
ax4.scatter(np.array(i_probs)[i_idx], np.log(np.array(i_ranks)[i_idx]), alpha=0.2, s=8, color='tomato', label='incorrect')
ax4.set_xlabel('Min Token Probability')
ax4.set_ylabel('Log Rank')
ax4.set_title('Probability vs Log Rank')
ax4.legend(markerscale=3)
fig4.tight_layout()
fig4.savefig('llm-probability-thresh/scatter_ppl_rank.png', dpi=150)

c_probs_arr = np.array(c_probs)
c_ranks_arr = np.array(c_ranks)
i_probs_arr = np.array(i_probs)
i_ranks_arr = np.array(i_ranks)

c_mask = c_probs_arr > 0.5
i_mask = i_probs_arr > 0.5

fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.scatter(c_probs_arr[c_mask], c_ranks_arr[c_mask], alpha=0.2, s=8, color='steelblue', label=f'correct (n={c_mask.sum()})')
ax5.scatter(i_probs_arr[i_mask], i_ranks_arr[i_mask], alpha=0.2, s=8, color='tomato', label=f'incorrect (n={i_mask.sum()})')
ax5.set_xlabel('Min Token Probability')
ax5.set_ylabel('Rank')
ax5.set_title('Probability vs Rank (prob > 0.5)')
ax5.legend(markerscale=3)
fig5.tight_layout()
fig5.savefig('llm-probability-thresh/scatter_highprob_rank.png', dpi=150)

plt.close('all')"""

