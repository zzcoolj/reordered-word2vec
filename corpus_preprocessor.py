# https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import time

lr = 0.05
dim = 200
ws = 5
epoch = 5
minCount = 5
neg = 5
loss = 'ns'
t = 1e-4

# Same values as used for fastText training above
params = {
    'alpha': lr,
    'size': dim,
    'window': ws,
    'iter': epoch,
    'min_count': minCount,
    'sample': t,
    'sg': 1,  # 1 for skip-gram
    'hs': 0,  # If 0, and negative is non-zero, negative sampling will be used.
    'negative': neg
}


def train_models(corpus_file, output_path):
    gs_model = Word2Vec(LineSentence(corpus_file), **params)
    gs_model.wv.save_word2vec_format(output_path)


start = time.time()
# train_models('input/enwiki-101M.txt', 'output/test')
train_models('input/enwiki-1G.txt', 'output/test1G')
end = time.time()
print('time (seconds):', end-start)