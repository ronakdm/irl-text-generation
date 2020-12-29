import random
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main(candidate_file):
	references = []
	with open("save/str_real_data.txt") as fin:
		for line in fin:
			line = line.split()
			references.append(line)
	random.shuffle(references)
	references = references[:1000]
	

	candidates = []
	with open(candidate_file)as fin:
		for line in fin:
			line = line.split()
			candidates.append(line)

	chencherry = SmoothingFunction()

	weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
	
	#for i in range(len(weights)):
		#weight = weights[i]
	for ngram in range(2, 6):
		weight = tuple((1. / ngram for _ in range(ngram)))
		scores = []
		for candidate in candidates:
			score = sentence_bleu(references, candidate, weights=weight, smoothing_function=chencherry.method1)
			scores.append(score)
		print(f"{len(weight)}-gram: ", sum(scores) / len(scores))


if __name__ == "__main__":
	main(sys.argv[1])
