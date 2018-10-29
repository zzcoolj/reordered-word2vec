# https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html

lr = 0.05
dim = 100
ws = 5
epoch = 5
minCount = 5
neg = 5
loss = 'ns'
t = 1e-4

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

# Same values as used for fastText training above
params = {
    'alpha': lr,
    'size': dim,
    'window': ws,
    'iter': epoch,
    'min_count': minCount,
    'sample': t,
    'sg': 1,
    'hs': 0,
    'negative': neg
}


def train_models(corpus_file, output_name):
    gs_model = Word2Vec(Text8Corpus(corpus_file), **params)
    gs_model.save_word2vec_format()
