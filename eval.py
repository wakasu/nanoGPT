import os
import numpy as np
import torch
from model import GPTConfig, GPT
# from model_mod1 import GPTConfig, GPT
import tiktoken
import math
from contextlib import nullcontext

def initialize_device(seed, device):
    """
    Initialize the device and set seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    return device_type, dtype

def get_batch(data_dir, dataset, split, batch_size, block_size, device_type, device):
    """
    Generate batches of data.
    """
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data_dir, dataset, batch_size, block_size, device_type, device, eval_iters):
    """
    Estimate the loss of the model.
    """
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type)
    out = {}
    model.eval()
    for split in ['val', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, dataset, split, batch_size, block_size, device_type, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = math.log2(losses.mean().item())
    return out

def load_model(ckpt_path, device, compile):
    """
    Load the model from a checkpoint.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    if compile:
        print("Compiling model...")
        model = torch.compile(model) # pytorch 2.0

    return model

def main():
    """
    Main function to run the evaluation.
    """
    batch_size = 12
    block_size = 1024
    eval_iters = 200
    seed = 1337
    device = 'cuda'
    compile = True
    data_dir='data/enwik8'
    dataset='enwik8'
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    # out_dir = 'out_3mods_2000iter'
    out_dir = 'out_org_2000iter'
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    compile = True # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read()) # overrides from command line or config file

    # data loading init
    dataset = 'enwik8'
    data_dir = os.path.join('data', dataset)

    exec(open('configurator.py').read())
    device_type, dtype = initialize_device(seed, device)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    model = load_model(ckpt_path, device=device, compile=compile)
    # losses = estimate_loss()
    losses = estimate_loss(model, data_dir, dataset, batch_size, block_size, device_type, device, eval_iters)

    print(f"test loss {losses['test']:.4f}, val loss {losses['val']:.4f}")

if __name__ == "__main__":
    main()