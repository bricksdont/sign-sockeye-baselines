import sys

testing_corpora = sys.argv[1].split(" ")

unseen_corpora = [t for t in testing_corpora if "unseen" in t]
unseen_corpora = [t.replace("_unseen", "") for t in unseen_corpora]

print(" ".join(unseen_corpora))
