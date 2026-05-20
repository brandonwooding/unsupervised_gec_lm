import csv
import re
from utils import tsv_to_dataset, login_to_huggingface, choose_device, load_tokenizer_and_model, pool_sub_tokens_to_words
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from torch.utils.data import DataLoader


## Load data
train_data_path = "multiGED-2023-english/en_fce_dev.tsv"
data = tsv_to_dataset(train_data_path)

label2id = {"c": 0, "i": 1}
data = data.map(lambda x: {"labels": [label2id[l] for l in x["labels"]]})


def collate_fn(batch):

    tokens = [item["tokens"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    encoding["word_labels"] = labels
    return encoding

dataloader = DataLoader(data, batch_size=32, collate_fn=collate_fn)
total_batches = len(dataloader)
total_words = sum(len(item["labels"]) for item in data)
feature_dir = Path("modernbert_word_features")
feature_dir.mkdir(exist_ok=True)

## Log in to hugging face
hf_token = login_to_huggingface(Path(".env"))

## Load model
device = choose_device("auto")
model_id = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, num_labels=2).to(device)
model.eval()

hidden_size = model.config.hidden_size
num_hidden_states = model.config.num_hidden_layers + 1

layer_arrays = [
    np.lib.format.open_memmap(
        feature_dir / f"layer_{layer_idx:02d}.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total_words, hidden_size),
    )
    for layer_idx in range(num_hidden_states)
]

labels_array = np.lib.format.open_memmap(
    feature_dir / "labels.npy",
    mode="w+",
    dtype=np.int64,
    shape=(total_words,),
)

write_pos = 0


with torch.inference_mode():
    for batch_idx, batch in enumerate(dataloader, start=1):
        print(f"batch {batch_idx}/{total_batches}")
        labels = batch.pop("word_labels")
        batch = batch.to(device)

        output = model(**batch, output_hidden_states=True)

        batch_word_labels = []
        for sentence_labels in labels:
            batch_word_labels.extend(sentence_labels)

        batch_word_count = len(batch_word_labels)
        batch_start = write_pos
        batch_end = batch_start + batch_word_count

        labels_array[batch_start:batch_end] = np.array(
            batch_word_labels,
            dtype=np.int64,
        )
        
        for layer_idx, layer in enumerate(output.hidden_states):
            layer_write_pos = batch_start

            for i in range(layer.shape[0]):
                word_ids = batch.word_ids(batch_index=i)

                word_vectors = pool_sub_tokens_to_words(layer[i], word_ids)

                assert len(labels[i]) == word_vectors.shape[0]

                n_words = word_vectors.shape[0]
                layer_arrays[layer_idx][
                    layer_write_pos:layer_write_pos + n_words
                ] = (
                    word_vectors.detach().cpu().half().numpy()
                )

                layer_write_pos += n_words

            assert layer_write_pos == batch_end

        write_pos = batch_end

        del output
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


assert write_pos == total_words

for layer_array in layer_arrays:
    layer_array.flush()
labels_array.flush()

metadata = {
    "label2id": label2id,
    "model_id": model_id,
    "total_words": total_words,
    "hidden_size": hidden_size,
    "num_hidden_states": num_hidden_states,
    "feature_dtype": "float16",
    "label_dtype": "int64",
}

torch.save(metadata, feature_dir / "metadata.pt")
