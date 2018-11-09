# https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html

from word2vec_modified import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

import time
import sys
sys.path.append('../word_embeddings_evaluator/')
from evaluator import Evaluator

lr = 0.05
dim = 200
ws = 5
epoch = 5
# minCount = 5
max_vocab_size = 20000
neg = 5
loss = 'ns'
t = 1e-4

# Same values as used for fastText training above
params = {
    'alpha': lr,
    'size': dim,
    'window': ws,
    'iter': epoch,
    # 'min_count': minCount,
    'max_vocab_size': max_vocab_size,
    'sample': t,
    'sg': 1,  # 1 for skip-gram
    'hs': 0,  # If 0, and negative is non-zero, negative sampling will be used.
    'negative': neg,
    'restricted_vocab': ['hello', 'world']
}


def train_models(corpus_file, output_path):
    gs_model = Word2Vec(LineSentence(corpus_file), **params)
    gs_model.wv.save_word2vec_format(output_path)


def evaluate(vec):
    # evaluation results
    labels1, results1 = Evaluator.evaluation_questions_words(vec)
    # self.print_lables_results(labels1, results1)
    labels2, results2 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/wordsim353/combined.tab')
    # eval.print_lables_results(labels2, results2)
    labels3, results3 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/simlex999.txt')
    # eval.print_lables_results(labels3, results3)
    return results2 + results3 + results1


start = time.time()
train_models('input/enwiki-101M.txt', 'output/test101M-vocab20000-restricted')
# train_models('input/enwiki-1G.txt', 'output/test1G-vocab20000')
end = time.time()
print('time (seconds):', end-start)


# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000').wv
# print(evaluate(vec))
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab20000').wv
# print(evaluate(vec))

