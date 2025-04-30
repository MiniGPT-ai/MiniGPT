import torch
from model import MiniGPTModel, MiniGPTConfig
from tokenizers import ByteLevelBPETokenizer

# Load config, tokenizer, model
config = MiniGPTConfig()
tokenizer = ByteLevelBPETokenizer("/MiniGPT-0/tokenizer/vocab.json", "/MiniGPT-0/tokenizer/merges.txt")
config.vocab_size = tokenizer.get_vocab_size()

model = MiniGPTModel(config)
model.load_state_dict(torch.load("/MiniGPT-0/minigpt0-preview.pth", map_location="cpu"))
model.eval()

# Sampling function
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long)

    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids[:, -config.block_size:])
            logits = logits[:, -1, :] / temperature

            if top_k:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            new_token = tokenizer.decode(next_token[0].tolist())
            print(new_token, end="", flush=True)

    print()


# Prompt
prompt = "What should I do when the user disables JavaScript on their browser?"
generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=10)