import tiktoken
import torch
from torch.nn import functional as F


@torch.no_grad()
def generate(model, prompt, device, max_length=50, top_k=50, num_return_sequences=4):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    B, T = num_return_sequences, tokens.size
    tokens = (
        torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
    )  # [B, T]
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    while T < max_length:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(tokens)  # [B, T, Vocab_size]
        logits = logits[:, -1, :]  # [B, T_last, Vocab_size]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=generator)  # (B, 1)
        next_token = torch.gather(
            topk_indices, -1, ix
        )  # gathers the appropriate  index of the selected prob, the index is the token
        tokens = torch.cat((tokens, next_token), dim=-1)

    for i in range(B):
        seq = tokens[
            i, :max_length
        ].tolist()  # for each sequence in the batch get the whole sequence
        decoded = enc.decode(seq)
        print(f"sample {i}: {decoded}")


# TODO:
# move the model from train_gpt2.py to model.py
# import model from model.py
# generate the model on cpu
