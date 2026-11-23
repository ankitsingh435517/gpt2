import torch
from model import GPT, GPTConfig

def create_small_model():
    config = GPTConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64)
    return GPT(config)


@torch.no_grad()
def test_output_shape():
    model = create_small_model()

    x = torch.randint(0, 256, (2, 16))  # 256 vocab_size, B = 2, T = 16

    logits,_ = model(x)  # (B, T, Vocab_size) = (2, 16, 256)

    assert logits.shape == (2, 16, 256)


@torch.no_grad()
def test_loss_is_finite():
    model = create_small_model()

    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    
    _, loss = model(x, y)

    assert loss is not None
    assert torch.isfinite(loss)


def test_backward_pass():
    model = create_small_model()

    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))

    logits, loss = model(x, y)
    
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
