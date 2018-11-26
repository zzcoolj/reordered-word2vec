# https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html

from word2vec_modified import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

import time
import sys
sys.path.append('../word_embeddings_evaluator/')
from evaluator import Evaluator

from scipy import spatial  # TODO NOW delete


def read_file_to_dict(file_path):
    l = []
    with open(file_path) as f:
        for line in f:
            l.append(line.rstrip('\n'))
    return dict.fromkeys(l, 1)


lr = 0.05
dim = 200
ws = 5
epoch = 5
# minCount = 5
max_vocab_size = 50000
neg = 5
loss = 'ns'
t = 1e-4
workers = 3  # 3 by default

restricted_vocab = read_file_to_dict('../word_embeddings_evaluator/data/distinct-tokens/353.txt')
restricted_type = 1

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
    'workers': workers,
    'restricted_vocab': restricted_vocab,  # [modified] ATTENTION: It must be a dictionary not a list!
    'restricted_type': restricted_type  # [modified] 0: train_batch_sg_original; 1: train_batch_sg_in; 2: train_batch_sg_notIn
}


def evaluate(vec):
    # evaluation results
    labels1, results1 = Evaluator.evaluation_questions_words(vec)
    # self.print_lables_results(labels1, results1)
    labels2, results2 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/wordsim353/combined.tab')
    # eval.print_lables_results(labels2, results2)
    labels3, results3 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/simlex999.txt')
    # eval.print_lables_results(labels3, results3)
    return results2 + results3 + results1


corpus_file = 'input/enwiki-1G.txt'
output_path = 'output/test1G-vocab50000-353test'
# corpus_file = '/Users/zzcoolj/Code/GoW/data/training data/Wikipedia-Dumps_en_20170420_prep/AA/wiki_01.txt'

start = time.time()
gs_model = Word2Vec(LineSentence(corpus_file), **params)
end = time.time()
print('1st step finished', 'time (seconds):', end-start)
print('again', gs_model.wv['again'][:10])
print('love', gs_model.wv['love'][:10])
print(evaluate(gs_model.wv))
gs_model.save(output_path)

# gs_model.restricted_type = 2
#
# print(evaluate(gs_model.wv))
#
# start = time.time()
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter)
# end = time.time()
# print('time (seconds):', end-start)
# print(evaluate(gs_model.wv))

# print('2nd step finished')
# print(gs_model.wv['love'])
# print(gs_model.wv['car'])
# result = 1 - spatial.distance.cosine(gs_model.wv['love'], gs_model.wv['car'])
# print(result)


# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-original').wv
# print(evaluate(vec))
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-999').wv
# print(evaluate(vec))
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-analogy').wv
# print(evaluate(vec))
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-analogy&353&999').wv
# print(evaluate(vec))
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-noTrain').wv
# print(evaluate(vec))
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-no353').wv
