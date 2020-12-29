import pickle

_, voc = pickle.load(open('save/vocab_cotra.pkl', 'rb'))

voc = {v: k for k,v in voc.items()}

sentences = open('save/real_data.txt', 'r').readlines()

res = open('save/str_real_data.txt', 'w')

for s in sentences:
    curr = ''
    words = s.split()

    for i in range(len(words)):
        if words[i] == '1814':
            break
        curr += voc[int(words[i])]
        curr += ' ' if i < len(words) - 1 and voc[int(words[i + 1])].isalnum() else ''

    # take out last space
    res.writelines(curr + '\n')

