import tiktoken
import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from pathlib import Path
import time
from torch.profiler import profile, ProfilerActivity, record_function

enc = tiktoken.get_encoding("gpt2")

@torch.no_grad()
def generate(
    model: GPT, prompt, device, max_length=50, top_k=50, num_return_sequences=4
):
    model.eval()
    with record_function("prompt_encoding"):
        tokens = enc.encode(prompt)
        tokens = (
            torch.tensor(tokens, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(num_return_sequences, 1)
        )  # [B, T]
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    while tokens.size(1) < max_length:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with record_function("model_forwarding"):
                logits, _ = model(tokens)  # [B, T, Vocab_size]
        logits = logits[:, -1, :]  # [B, T_last, Vocab_size]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=generator)  # (B, 1)
        next_token = torch.gather(
            topk_indices, -1, ix
        )  # gathers the appropriate  index of the selected prob, the index is the token
        tokens = torch.cat((tokens, next_token), dim=-1)

    for i in range(num_return_sequences):
        seq = tokens[
            i, :max_length
        ].tolist()  # for each sequence in the batch get the whole sequence
        decoded = enc.decode(seq)
        print(f"sample {i}: {decoded}")


def generateSequences(prompts):
    device = "cpu"
    model = GPT(GPTConfig(vocab_size=50304))
    BASE_DIR = Path(__file__).resolve().parent
    checkpoint_path = BASE_DIR / "checkpoints" / "gpt2.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Will remove it after next training run.
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint)
    model.to(device)

    for i in range(len(prompts)):
        t0 = time.time()
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            generate(model, prompts[i], device, max_length=50, num_return_sequences=1)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace("traces/cpu_inference_trace.json")
        t1 = time.time()
        dt = t1 - t0
        dt_ms = dt * 1000
        print(f"Time: {dt_ms}")
        print("\n")


prompts = [
    "On a cold winter morning, a young man walked into an abandoned railway station and noticed something strange about the clock.",
]
generateSequences(prompts)

# baseline: tokens: 50 time: 7s (avg)
# TODO: reduce the time for the baseline inference
