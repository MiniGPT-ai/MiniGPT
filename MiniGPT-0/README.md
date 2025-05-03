![preview](https://img.shields.io/badge/status-research--preview-orange.svg)

# MiniGPT-0 (Research Preview)

This is the research preview of **MiniGPT-0**, a lightweight GPT-style transformer model with ~7.3 million parameters.

> ⚠️ This model is intended **for research and educational use only**.  
> It is **not aligned** or safety-tested for deployment.  
> See `WEIGHTS_LICENSE.md` for usage restrictions.

---

## Model Details

| Field           | Value        |
|----------------|--------------|
| Parameters      | ~7.3M        |
| Layers          | 6            |
| Heads           | 4            |
| Embedding Size  | 256          |
| FFN Size        | 1024         |
| Block Size      | 128 tokens   |
| Vocab Size      | 5000         |
| Dropout         | 0.1          |

---

## Files

- `model.py` – model architecture  
- `config.json` – model config file  
- `train.py` – training script  
- `generate.py` – generation script  
- `tokenizer/` – trained BPE tokenizer (`vocab.json`, `merges.txt`)  
- `minigpt0-preview.pth` – pretrained weights  
- `WEIGHTS_LICENSE.md` – research-only use license  

---

## Example Usage

```python
from model import MiniGPTModel, MiniGPTConfig
import torch

config = MiniGPTConfig("config.json")
model = MiniGPTModel(config)
model.load_state_dict(torch.load("minigpt0-preview.pth", map_location="cpu"))
model.eval()

from tokenizer import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")


## License

- **Code**: MIT License (see `LICENSE`)
- **Pretrained Weights**: Research-only license (see `WEIGHTS_LICENSE.md`)
