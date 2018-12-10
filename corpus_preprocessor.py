# https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html

from word2vec_modified import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

import time
import pickle
import pandas as pd
import sys
sys.path.append('../word_embeddings_evaluator/')
sys.path.append('../common/')
from evaluator import Evaluator
from common import write_to_pickle, read_pickle


def read_file_to_dict(file_path):
    l = []
    with open(file_path) as f:
        for line in f:
            l.append(line.rstrip('\n'))
    return dict.fromkeys(l, 1)


def evaluate(vec, name):
    # evaluation results
    labels1, results1 = Evaluator.evaluation_questions_words(vec)
    # self.print_lables_results(labels1, results1)
    labels2, results2 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/wordsim353/combined.tab')
    # eval.print_lables_results(labels2, results2)
    labels3, results3 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/simlex999.txt')
    # eval.print_lables_results(labels3, results3)
    labels4, results4 = Evaluator.evaluation_word_pairs(vec, evaluation_data_path='~/Code/word_embeddings_evaluator/data/MTURK-771.csv', delimiter=',')

    return [name] + results2 + results3 + results4 + results1


def alpha_splitter(start, epochs, end=0.0001):
    alphas = list()
    alphas.append(start)
    for i in range(1, epochs):
        alpha = start - ((start - end) * float(i) / epochs)
        alphas.append(alpha)
    alphas.append(end)
    return alphas


def iteration_simulater(total_epoch, special_epoch_count, restricted_vocab_name):
    # corpus_file = '/Users/zzcoolj/Code/GoW/data/training data/Wikipedia-Dumps_en_20170420_prep/AA/wiki_01.txt'
    corpus_file = 'input/enwiki-1G.txt'
    xlsx_path = 'output/test1G-vocab50000-original-iter' + str(total_epoch) + '-lastEpochInitial-' + \
                str(restricted_vocab_name) + '.xlsx'
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
    line_number_in_xlsx = 0

    # epoch 0
    lr = 0.025
    alphas = alpha_splitter(start=lr, epochs=total_epoch)
    print('alphas', alphas)
    min_alpha = alphas[1]
    restricted_vocab = read_file_to_dict('../word_embeddings_evaluator/data/distinct-tokens/' +
                                         str(restricted_vocab_name) + '.txt')
    restricted_type = 0
    params = {
        'alpha': lr,
        'min_alpha': min_alpha,
        'size': 200,
        'window': 5,
        'iter': 1,
        'max_vocab_size': 50000,
        'sample': 1e-4,
        'sg': 1,  # 1 for skip-gram
        'hs': 0,  # If 0, and negative is non-zero, negative sampling will be used.
        'negative': 5,
        'workers': 3,

        'restricted_vocab': restricted_vocab,  # [modified] ATTENTION: It must be a dictionary not a list!
        'restricted_type': restricted_type  # [modified] 0: train_batch_sg_original; 1: train_batch_sg_in; 2: train_batch_sg_notIn
    }
    print('cur_epoch', 0)
    gs_model = Word2Vec(LineSentence(corpus_file), **params)
    df.loc[line_number_in_xlsx] = evaluate(gs_model.wv, 'epoch0')

    # # epoch 0.5
    # gs_model.restricted_type = 2
    # gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter,
    #                start_alpha=lr, end_alpha=min_alpha)
    # df.loc[1] = evaluate(gs_model.wv, 'X-iter0.5')

    # epoch 1+
    # gs_model.restricted_type = 0
    for cur_epoch in range(1, total_epoch-special_epoch_count):
        print('cur_epoch', cur_epoch)
        start_alpha = alphas[cur_epoch]
        end_alpha = alphas[cur_epoch+1]
        print('start_alpha', start_alpha)
        print('end_alpha', end_alpha)
        gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter,
                       start_alpha=start_alpha, end_alpha=end_alpha)
        line_number_in_xlsx += 1
        df.loc[line_number_in_xlsx] = evaluate(gs_model.wv, 'epoch'+str(cur_epoch))

    # save common base model
    write_to_pickle(gs_model, xlsx_path.split('.xlsx')[0]+'-base')

    for special_epoch in range(total_epoch-special_epoch_count, total_epoch):
        print('special epoch', special_epoch)
        start_alpha = alphas[special_epoch]
        end_alpha = alphas[special_epoch+1]
        print('start_alpha', start_alpha)
        print('end_alpha', end_alpha)
        # final special epochs 0.5
        gs_model.restricted_type = 1
        gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter,
                       start_alpha=start_alpha, end_alpha=end_alpha)
        line_number_in_xlsx += 1
        df.loc[line_number_in_xlsx] = evaluate(gs_model.wv, 'epoch'+str(special_epoch)+'-half')

        # final special epochs final
        gs_model.restricted_type = 2
        gs_model.train(LineSentence(corpus_file), total_examples=gs_model.corpus_count, epochs=gs_model.iter,
                       start_alpha=start_alpha, end_alpha=end_alpha)
        line_number_in_xlsx += 1
        df.loc[line_number_in_xlsx] = evaluate(gs_model.wv, 'epoch' + str(special_epoch)+'-entire')

    # baseline (final original word2vec epochs)
    gs_model_base = read_pickle(xlsx_path.split('.xlsx')[0] + '-base')
    gs_model_base.restricted_type = 0
    for baseline_epoch in range(total_epoch - special_epoch_count, total_epoch):
        print('baseline epoch', baseline_epoch)
        start_alpha = alphas[baseline_epoch]
        end_alpha = alphas[baseline_epoch + 1]
        print('start_alpha', start_alpha)
        print('end_alpha', end_alpha)
        gs_model_base.train(LineSentence(corpus_file), total_examples=gs_model_base.corpus_count, epochs=gs_model_base.iter,
                            start_alpha=start_alpha, end_alpha=end_alpha)
        line_number_in_xlsx += 1
        df.loc[line_number_in_xlsx] = evaluate(gs_model_base.wv, 'epoch' + str(baseline_epoch)+'-baseline')

    writer = pd.ExcelWriter(xlsx_path)
    df.to_excel(writer, 'Sheet1')
    writer.save()


""" Local test """
# corpus_file = '/Users/zzcoolj/Code/GoW/data/training data/Wikipedia-Dumps_en_20170420_prep/AA/wiki_01.txt'
# Word2Vec(LineSentence(corpus_file), **params)


""" Evaluate Embeddings """
# vec = KeyedVectors.load_word2vec_format('output/test1G-vocab50000-original').wv
# evaluate(vec, 'output/test1G-vocab50000-original')


""" Normal word embeddings training """
# corpus_file = 'input/enwiki-1G.txt'
# xlsx_path = 'output/test1G-vocab50000-original-alpha.xlsx'
# lrs = [0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.01, 0.005]
#
# df = pd.DataFrame(columns=[
#     # word embeddings file name
#     'file name',
#     # wordsim353
#     'wordsim353_Pearson correlation', 'Pearson pvalue',
#     'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
#     # simlex999
#     'simlex999_Pearson correlation', 'Pearson pvalue',
#     'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
#     # MTURK-771
#     'MTURK771_Pearson correlation', 'Pearson pvalue',
#     'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
#     # questions-words
#     'sem_acc', '#sem', 'syn_acc', '#syn', 'total_acc', '#total'
# ])
#
# i = 0
#
# for lr in lrs:
#     print(lr)
#     params = {
#         'alpha': lr,
#         'min_alpha': 0.0001,
#         'size': 200,
#         'window': 5,
#         'iter': 5,
#         'max_vocab_size': 50000,
#         'sample': 1e-4,
#         'sg': 1,  # 1 for skip-gram
#         'hs': 0,  # If 0, and negative is non-zero, negative sampling will be used.
#         'negative': 5,
#         'workers': 3,
#
#         'restricted_vocab': None,  # [modified] ATTENTION: It must be a dictionary not a list!
#         'restricted_type': 0  # [modified] 0: train_batch_sg_original; 1: train_batch_sg_in; 2: train_batch_sg_notIn
#     }
#     gs_model = Word2Vec(LineSentence(corpus_file), **params)
#     df.loc[i] = evaluate(gs_model.wv, str(lr))
#     i += 1
#
# writer = pd.ExcelWriter(xlsx_path)
# df.to_excel(writer, 'Sheet1')
# writer.save()


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
