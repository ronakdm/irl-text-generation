import pickle
from transformers import GPT2Tokenizer

tk = GPT2Tokenizer.from_pretrained("gpt2")

# load old tokens
_, voc = pickle.load(open("save/vocab_cotra.pkl", "rb"))

voc_map = dict()  # do not include space.

gpt_map = []
str_map = []

i = 0
for word in voc.keys():
    # pad never used and space is taken care of by tokenizer
    if word != " " and word != "PADTOKEN":
        voc_map[word] = i
        i += 1
        gpt_map.append(tk.encode(word))
        str_map.append(word)  # reverse map
        # voc_map.append(tokenizer.encode(word))
    else:
        print(word)

voc_map[tk.eos_token] = i  # <eos>
str_map.append(tk.eos_token)
gpt_map.append(50256)

pickle.dump(gpt_map, open("save/gpt_map.pkl", "wb"))
pickle.dump(voc_map, open("save/str_vocab_map.pkl", "wb"))
pickle.dump(str_map, open("save/str_map.pkl", "wb"))
