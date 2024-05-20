"""
This example computes the score between a query and all possible
sentences in a corpus using a Cross-Encoder for semantic textual similarity (STS).
It output then the most similar sentences for the given query.
"""

from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

# Pre-trained cross encoder
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# We want to compute the similarity between the query sentence
query = "A man is eating pasta."

# With all sentences in the corpus
corpus = [
    "A man is eating pasta.",
    "A rat is eating noodles.",
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# 1. We rank all sentences in the corpus for the query
ranks = model.rank(query, corpus)

# Print the scores
print("Query:", query)
print('------------------------------------')
for rank in ranks:
    print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")
