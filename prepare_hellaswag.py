import os
from datasets import load_dataset

output_dir = "/mnt/gpt2/hellaswag"
output_file = os.path.join(output_dir, "hellaswag_val.jsonl")

os.makedirs(output_dir, exist_ok=True)

print("Downloading HellaSwag from Hugging Face...")

dataset = load_dataset(
    "Rowan/hellaswag",
    split="validation"
)

print("Number of examples:", len(dataset))

temp_file = output_file + ".tmp"

dataset.to_json(
    temp_file,
    orient="records",
    lines=True
)

os.replace(temp_file, output_file)

print("Saved:", output_file)
print("Dataset preparation complete!")