import os

TOKENS_FILE = os.environ["DATA_PATH"] + "/interim/articles/tokens.pickle"



model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")