import os
from gensim.models import Word2Vec, KeyedVectors
import src
W2V_FILE = os.environ["MODEL_PATH"] + "/word2vec.model"

model = KeyedVectors.load(os.environ["MODEL_PATH"] + "/word2vec_kv.model")

print(len(model))