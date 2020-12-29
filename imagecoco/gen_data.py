import pickle
from transformers import GPT2Tokenizer
import torch
import re

SEQ_LEN = 32

# Load word->voc map
word_voc_map = pickle.load(open("save/str_vocab_map.pkl", "rb"))

# get gpt2 tok
tk = GPT2Tokenizer.from_pretrained("gpt2")

# load in sentence data
sentences = open("save/str_real_data.txt", "r")
str_data = [line.rstrip("\n") for line in sentences]

# define outputs
tok_data = []  # tokenized data
gpt_tok_data = []  # gpt2 tok data
tok_mask = []  # for masking out (since can map to > 1)


for i in range(len(str_data)):
    # split
    s_spl = re.findall(r"\w+|[^\w\s]", str_data[i])

    """
        Tokenize into new vocab
    """
    # tokenize each part
    tok_data.append([word_voc_map[w] for w in s_spl])

    """
        Tokenize each token from previous step into GPT2
        token_mask=1 at last token for each word
    """
    tok = tk(s_spl)["input_ids"]

    mask = []
    tok_f = []
    for t in tok:
        mask += (len(t) - 1) * [False] + [True]
        tok_f.extend(t)
    tok_mask.append(mask)
    gpt_tok_data.append(tok_f)


# find max len for padding
max_tok = max([len(x) for x in tok_data])
max_gpt_tok = max([len(x) for x in gpt_tok_data])

# pad
for i in range(len(str_data)):
    tok_data[i] += (max_tok - len(tok_data[i])) * [word_voc_map["eos"]]

    gpt_tok_data[i] += (max_gpt_tok - len(tok_mask[i])) * [50256]
    tok_mask[i] += (max_gpt_tok - len(tok_mask[i])) * [True]

# conv to tensors
tok_data = torch.tensor(tok_data)
gpt_tok_data = torch.tensor(gpt_tok_data)
tok_mask = torch.tensor(tok_mask)

# dump into file
pickle.dump((tok_data, gpt_tok_data, tok_mask), open("save/train_data.pkl", "wb"))
