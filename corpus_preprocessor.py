# https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html

from word2vec_modified import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

import time
import pandas as pd
import sys
sys.path.append('../word_embeddings_evaluator/')
from evaluator import Evaluator


def read_file_to_dict(file_path):
    l = []
    with open(file_path) as f:
        for line in f:
            l.append(line.rstrip('\n'))
    return dict.fromkeys(l, 1)


lr = 0.05
dim = 200
ws = 5
epoch = 5  # TODO NOW
# minCount = 5
max_vocab_size = 50000
neg = 5
loss = 'ns'
t = 1e-4
workers = 3  # 3 by default

# restricted_vocab = read_file_to_dict('../word_embeddings_evaluator/data/distinct-tokens/analogy&353&999.txt')
restricted_vocab = read_file_to_dict('../word_embeddings_evaluator/data/distinct-tokens/353.txt')  # TODO NOW
restricted_type = 0  # TODO NOW

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


def evaluate(vec, output_path):
    # evaluation results
    labels1, results1 = Evaluator.evaluation_questions_words(vec)
    # self.print_lables_results(labels1, results1)
    labels2, results2 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/wordsim353/combined.tab')
    # eval.print_lables_results(labels2, results2)
    labels3, results3 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/simlex999.txt')
    # eval.print_lables_results(labels3, results3)
    labels4, results4 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/MTURK-771.csv', delimiter=',')

    df = pd.DataFrame(columns=[
        # word embeddings file name
        'file name',
        # wordsim353
        'wordsim353_Pearson correlation', 'Pearson pvalue',
        'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
        # simlex999
        'simlex999_Pearson correlation', 'Pearson pvalue',
        'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
        # MTURK-771
        'MTURK771_Pearson correlation', 'Pearson pvalue',
        'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
        # questions-words
        'sem_acc', '#sem', 'syn_acc', '#syn', 'total_acc', '#total'
    ])

    df.loc[0] = [output_path] + results2 + results3 + results4 + results1
    output_path += '.xlsx'
    writer = pd.ExcelWriter(output_path)
    df.to_excel(writer, 'Sheet1')
    writer.save()


""" Local test """
corpus_file = '/Users/zzcoolj/Code/GoW/data/training data/Wikipedia-Dumps_en_20170420_prep/AA/wiki_01.txt'
Word2Vec(LineSentence(corpus_file), **params)


""" Evaluate Embeddings """
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-original').wv
# evaluate(vec, 'output/test1G-vocab50000-original')


""" Separate Training """
# for end_alpha in [0.0005, 0.001, 0.005, 0.01, 0.025]:  # original: 0.0001
#     corpus_file = 'input/enwiki-1G.txt'
#     # output_path = 'output/test1G-vocab50000-noAnalogy&353&999-analogy&353&999test'
#     output_path = 'output/test1G-vocab50000-no353-353-' + str(end_alpha)
#     # corpus_file = '/Users/zzcoolj/Code/GoW/data/training data/Wikipedia-Dumps_en_20170420_prep/AA/wiki_01.txt'
#     # start_alpha=gs_model.min_alpha_yet_reached
#
#     print(output_path)
#
#     start = time.time()
#     gs_model = Word2Vec(LineSentence(corpus_file), **params)
#     end = time.time()
#     print('1st step finished', 'time (seconds):', end-start)
#     # print('again', gs_model.wv['again'][:10])
#     # print('go', gs_model.wv['go'][:10])
#     # print('love', gs_model.wv['love'][:10])
#     print(evaluate(gs_model.wv))
#     # gs_model.save(output_path)
#
#     gs_model.restricted_type = 1
#     start = time.time()
#     gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter,
#                    end_alpha=end_alpha)
#     end = time.time()
#     print('2nd step finished', 'time (seconds):', end-start)
#     # print('again', gs_model.wv['again'][:10])
#     # print('go', gs_model.wv['go'][:10])
#     # print('love', gs_model.wv['love'][:10])
#     print(evaluate(gs_model.wv))
#     gs_model.save(output_path)


""" Entire Training """
# corpus_file = 'input/enwiki-1G.txt'
# output_path = 'output/test1G-vocab50000-no999-999-entire-alphaControl'
# print(output_path)
# gs_model = Word2Vec(LineSentence(corpus_file), **params)
# gs_model.restricted_type = 1
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# print(evaluate(gs_model.wv))
#
# gs_model.restricted_type = 2
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# gs_model.restricted_type = 1
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# print(evaluate(gs_model.wv))
#
# gs_model.restricted_type = 2
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# gs_model.restricted_type = 1
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# print(evaluate(gs_model.wv))
#
# gs_model.restricted_type = 2
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# gs_model.restricted_type = 1
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# print(evaluate(gs_model.wv))
#
# gs_model.restricted_type = 2
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# gs_model.restricted_type = 1
# print(gs_model.min_alpha_yet_reached)
# gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter, start_alpha=gs_model.min_alpha_yet_reached)
# print(evaluate(gs_model.wv))
#
# gs_model.save(output_path)
