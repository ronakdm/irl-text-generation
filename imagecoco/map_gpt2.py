import pickle
from transformers import GPT2Tokenizer
import torch

sentences = open("save/str_real_data.txt", "r")
sentences = [line.rstrip("\n") for line in sentences]

tk = GPT2Tokenizer.from_pretrained("gpt2", padding=True)
tk.pad_token = tk.eos_token

res = tk(sentences, padding=True)
tok = torch.tensor(res["input_ids"])
attn_mask = torch.tensor(res["attention_mask"])

# binary mask
tok_mask = torch.zeros(tok.shape)

for i, s in enumerate(sentences):
    curr = 0
    words = s.split()

    t = tk(words)["input_ids"]
    a = []
    for token in t:
        a.extend(token)

    a = torch.tensor(a + (34 - len(a)) * [50256])
    print(words)
    print(a)
    print(tok[i])

    assert a.eq(tok[i]), "NOPE"

    for w in words:
        try:
            num_tok = len(tk.encode(w))
            curr += num_tok

            tok_mask[i][curr - 1] = 1
        except Exception:
            print(len(tok[i]))

pickle.dump((tok, attn_mask, tok_mask), open("save/str_gpt_data.pkl", "wb"))
