# -*- encoding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict
from joblib import Parallel, delayed
import cPickle as pkl
from nn.utils.io_utils import serialize_to_file, deserialize_from_file
import dill
import os

ENABLE_NULL_ALIGNMENT = False
LOAD_PARAMS = True


def argmax_j(f, e_i, theta):
    max_prob = -9999.
    max_j = -1

    for j, f_j in enumerate(f):
        prob = theta[(e_i, f_j)]
        if prob > max_prob:
            max_prob = prob
            max_j = j

    return max_j, max_prob


def argmax_i(e, f_j, theta):
    max_prob = -9999.
    max_i = -1

    for i, e_i in enumerate(e):
        prob = theta[(e_i, f_j)]
        if prob > max_prob:
            max_prob = prob
            max_i = i

    return max_i, max_prob


class IBMModel1(object):
    def __init__(self, bitext, src_vocab, tgt_vocab, max_iter=10, dir='f2e'):
        self.bitext = bitext
        self.max_iter = max_iter
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.epsilon = 1. / max(len(src_sent) for src_sent, tgt_sent in bitext)
        self.theta = None
        self.dir = dir

    def train(self):
        print('start training... Source Vocab[%d], Target Vocab[%d]' % (len(self.src_vocab), len(self.tgt_vocab)))

        # initialize p(f_j|e_i)

        # p_f_e_init = 1. / len(self.src_vocab)
        # theta = defaultdict(lambda: p_f_e_init)

        theta = dict()
        counts_e = defaultdict(lambda: set())
        for src_sent, tgt_sent in self.bitext:
            src_sent = [self.src_vocab[w] for w in src_sent]
            tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]

            for f_j in src_sent:
                for e_i in tgt_sent:
                    theta[(e_i, f_j)] = 0.
                    counts_e[e_i].add(f_j)

        for e_i, f_j in theta:
            theta[(e_i, f_j)] = 1. / len(counts_e[e_i])

        del counts_e

        for t in xrange(self.max_iter):
            # E step
            counts_e = defaultdict(float)
            counts = defaultdict(float)

            for src_sent, tgt_sent in self.bitext:
                src_sent = [self.src_vocab[w] for w in src_sent]
                tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]

                for j in xrange(len(src_sent)):
                    f_j = src_sent[j]
                    marginal = sum(theta[(e_i, f_j)] for e_i in tgt_sent)

                    for i in xrange(len(tgt_sent)):
                        e_i = tgt_sent[i]
                        align_posterior = theta[(e_i, f_j)] / marginal
                        counts[(e_i, f_j)] += align_posterior
                        counts_e[e_i] += align_posterior

            # M step
            for e_i, f_j in counts:
                if counts_e[e_i] > 0.:
                    theta[(e_i, f_j)] = counts[(e_i, f_j)] / counts_e[e_i]

            # compute log-likelihood
            ll = 0.
            for src_sent, tgt_sent in self.bitext:
                src_sent = [self.src_vocab[w] for w in src_sent]
                tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]

                p_F_given_E = np.log(self.epsilon) - len(src_sent) * np.log(len(tgt_sent))
                # p_F_given_E = - len(src_sent) * np.log(len(tgt_sent))
                for f_j in src_sent:
                    p_fj = 0.
                    for e_i in tgt_sent:
                        p_fj += theta[(e_i, f_j)]

                    p_F_given_E += np.log(p_fj)

                ll += p_F_given_E

            ll /= sum(len(src_sent) for src_sent, tgt_sent in self.bitext)
            print('avg. ll per word: %f' % ll)

            self.theta = theta

    def align(self):
        alignments = []
        for idx, (f, e) in enumerate(self.bitext):
            if self.dir == 'e2f':
                t = f; f = e; e = t

            cur_alignments = []
            for j in xrange(len(f)):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)

                f_j = f[j]

                max_prob = -9999.
                max_i = -1
                for i, e_i in enumerate(e):
                    if self.dir == 'e2f':
                        # magical hack that nobody knows why ...
                        prob = self.theta[(self.tgt_vocab[f_j], self.src_vocab[e_i])]
                    else:
                        prob = self.theta[(self.tgt_vocab[e_i], self.src_vocab[f_j])]

                    if prob > max_prob:
                        max_prob = prob
                        max_i = i

                if ENABLE_NULL_ALIGNMENT and max_i == len(e) - 1:
                    continue # skip the last null word in E

                cur_alignments.append((j, max_i))

            alignments.append(cur_alignments)

        return alignments


class IBMModel2(object):
    def __init__(self, bitext, theta, src_vocab, tgt_vocab, max_iter=10, dir='f2e'):
        self.bitext = bitext
        self.max_iter = max_iter
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.theta = theta
        self.align_prob = None
        self.epsilon = 1. / max(len(src_sent) for src_sent, tgt_sent in bitext)
        self.dir = dir

    @staticmethod
    def bucketize(e_len, f_len):
        # return 1, 1
        return e_len, f_len
        # def _bucketize(l):
        #     if l < 10:
        #         return (l / 3) * 3
        #     elif l < 50:
        #         return (l / 10) * 10
        #     else:
        #         return 50

        # return _bucketize(e_len), _bucketize(f_len)

    def train(self):
        print('start training IBM Model 2... Source Vocab[%d], Target Vocab[%d]' % (len(self.src_vocab), len(self.tgt_vocab)))

        # initialize alignment probabilities
        align_prob = defaultdict(float)

        for src_sent, tgt_sent in self.bitext:
            e_len = len(tgt_sent)
            f_len = len(src_sent)

            e_len_key, f_len_key = self.bucketize(e_len, f_len)
            for j in xrange(f_len):
                for i in xrange(e_len):
                    align_prob[(i, j, f_len_key, e_len_key)] = 1. / e_len

        theta = self.theta

        for t in xrange(self.max_iter):
            # E step
            counts_f_given_e = defaultdict(float)
            counts_e = defaultdict(float)

            counts_i_given_j = defaultdict(float)
            counts_j = defaultdict(float)

            for src_sent, tgt_sent in self.bitext:
                src_sent = [self.src_vocab[w] for w in src_sent]
                tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]

                f_len = len(src_sent)
                e_len = len(tgt_sent)
                e_len_key, f_len_key = self.bucketize(e_len, f_len)

                for j in xrange(f_len):
                    f_j = src_sent[j]
                    f_j_subtotal = sum(theta[(e_i, f_j)] * align_prob[(i, j, f_len_key, e_len_key)] for i, e_i in enumerate(tgt_sent))

                    for i in xrange(e_len):
                        e_i = tgt_sent[i]
                        c = theta[(e_i, f_j)] * align_prob[(i, j, f_len_key, e_len_key)] / f_j_subtotal
                        counts_f_given_e[(e_i, f_j)] += c
                        counts_e[e_i] += c
                        counts_i_given_j[(i, j, f_len_key, e_len_key)] += c
                        counts_j[(j, f_len_key, e_len_key)] += c

            # M step
            for e_i, f_j in counts_f_given_e:
                theta[(e_i, f_j)] = counts_f_given_e[(e_i, f_j)] / counts_e[e_i]

            for i, j, f_len, e_len in counts_i_given_j:
                align_prob[(i, j, f_len, e_len)] = counts_i_given_j[(i, j, f_len, e_len)] / counts_j[(j, f_len, e_len)]

            # compute log-likelihood
            ll = 0.
            for src_sent, tgt_sent in self.bitext:
                src_sent = [self.src_vocab[w] for w in src_sent]
                tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]

                e_len = len(tgt_sent)
                f_len = len(src_sent)
                e_len_key, f_len_key = self.bucketize(e_len, f_len)

                p_F_given_E = np.log(self.epsilon)
                for j, f_j in enumerate(src_sent):
                    p_fj = 0.
                    for i, e_i in enumerate(tgt_sent):
                        p_fj += theta[(e_i, f_j)] * align_prob[(i, j, f_len_key, e_len_key)]

                    p_F_given_E += np.log(p_fj)

                ll += p_F_given_E / len(src_sent)

            ll /= len(self.bitext)
            print('avg. ll per word: %f' % ll)

            self.theta = theta
            self.align_prob = align_prob

    def align(self):
        alignments = []
        for idx, (f, e) in enumerate(self.bitext):
            if self.dir == 'e2f':
                t = f; f = e; e = t

            cur_alignments = []
            e_len = len(e)
            f_len = len(f)
            e_len_key, f_len_key = self.bucketize(e_len, f_len)

            for j in xrange(len(f)):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)

                f_j = f[j]

                max_prob = -9999.
                max_i = -1
                for i, e_i in enumerate(e):
                    if self.dir == 'e2f':
                        prob = self.theta[(self.tgt_vocab[f_j], self.src_vocab[e_i])] # * self.align_prob[(j, i, e_len_key, f_len_key)]
                    else:
                        prob = self.theta[(self.tgt_vocab[e_i], self.src_vocab[f_j])] # * self.align_prob[(i, j, f_len_key, e_len_key)]

                    if prob > max_prob:
                        max_prob = prob
                        max_i = i

                if ENABLE_NULL_ALIGNMENT and max_i == len(e) - 1:
                    continue # skip the last null word in E

                cur_alignments.append((j, max_i))

            alignments.append(cur_alignments)

        return alignments


def grow_diag_final_and(src_len, tgt_len, e2f_alignments, f2e_alignments):
    """
    adapted from philipp koehn's book
    """
    neighboring = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    f_aligned = set()
    e_aligned = set()

    alignments = set(e2f_alignments).intersection(set(f2e_alignments))
    union_alignments = set(e2f_alignments).union(set(f2e_alignments))

    for j, i in alignments:
        f_aligned.add(j)
        e_aligned.add(i)

    def grow_diag():
        has_new = True
        while has_new:
            has_new = False
            for e_i in xrange(tgt_len):
                for f_j in xrange(src_len):
                    if (f_j, e_i) in alignments:
                        for neighbor in neighboring:
                            f_j_prime = neighbor[0] + f_j
                            e_i_prime = neighbor[1] + e_i

                            if (f_j_prime not in f_aligned and e_i_prime not in e_aligned) \
                                and (f_j_prime, e_i_prime) in union_alignments:
                                alignments.add((f_j_prime, e_i_prime))
                                e_aligned.add(e_i_prime)
                                f_aligned.add(f_j_prime)
                                has_new = True

    def final(a):
        for e_i in xrange(tgt_len):
            for f_j in xrange(src_len):
                if (f_j not in f_aligned and e_i not in e_aligned) and (f_j, e_i) in a:
                    alignments.add((f_j, e_i))
                    f_aligned.add(f_j)
                    e_aligned.add(e_i)

    grow_diag()
    final(e2f_alignments)
    final(f2e_alignments)

    return alignments


def train_alignment(data):
    bitext, src_vocab, tgt_vocab, dir = data
    model1 = IBMModel1(bitext, src_vocab, tgt_vocab, max_iter=10, dir=dir)
    model1.train()
    alignments_model1 = model1.align()

    model1_f = os.path.join(os.path.split(sys.argv[3])[0], 'model1.%s.bin' % dir)
    serialize_to_file(model1.theta, model1_f)
    print('saved model1 parameters to: %s' % model1_f)

    model2 = IBMModel2(bitext, model1.theta, src_vocab, tgt_vocab, dir=dir)
    model2.train()
    alignments_model2 = model2.align()

    model2_f = os.path.join(os.path.split(sys.argv[3])[0], 'model2.%s.bin' % dir)
    serialize_to_file([model2.theta, model2.align_prob], model2_f)
    print('saved model2 parameters to: %s' % model2_f)

    return alignments_model1, alignments_model2


if __name__ == '__main__':
    src_vocab = defaultdict(lambda: len(src_vocab))
    tgt_vocab = defaultdict(lambda: len(tgt_vocab))

    if ENABLE_NULL_ALIGNMENT:
        src_vocab['<null>'] = 0
        tgt_vocab['<null>'] = 0

    bitext = []
    for src_sent, tgt_sent in zip(open(sys.argv[1]), open(sys.argv[2])):
        src_words = src_sent.strip().split(' ')
        tgt_words = tgt_sent.strip().split(' ')

        for src_word in src_words:
            wid = src_vocab[src_word]
        for tgt_word in tgt_words:
            wid = tgt_vocab[tgt_word]

        bitext.append((src_words, tgt_words))

    print('num. bitext: %d' % len(bitext))

    if ENABLE_NULL_ALIGNMENT:
        fe_bitext = [(src_sent, tgt_sent + ['<null>']) for src_sent, tgt_sent in bitext]
        ef_bitext = [(tgt_sent, src_sent + ['<null>']) for src_sent, tgt_sent in bitext]
    else:
        fe_bitext = [(src_sent, tgt_sent) for src_sent, tgt_sent in bitext]
        ef_bitext = [(tgt_sent, src_sent) for src_sent, tgt_sent in bitext]

    if len(sys.argv[1:]) == 3:
        alignments = Parallel(n_jobs=2)(delayed(train_alignment)(data) for data in
                                        [(fe_bitext, src_vocab, tgt_vocab, 'f2e'),
                                         (ef_bitext, tgt_vocab, src_vocab, 'e2f')])

        (fe_alignments_model1, fe_alignments_model2), (ef_alignments_model1, ef_alignments_model2) = alignments

        # fe_model1 = IBMModel1(fe_bitext, src_vocab, tgt_vocab, max_iter=10)
        # fe_model1.train()
        # fe_alignments_model1 = fe_model1.align()

        # ef_model1 = IBMModel1(ef_bitext, tgt_vocab, src_vocab, max_iter=10)
        # ef_model1.train()
        # ef_alignments_model1 = ef_model1.align()

        # fe_model2 = IBMModel2(fe_bitext, fe_model1.theta, src_vocab, tgt_vocab, max_iter=10)
        # fe_model2.train()
        # fe_alignments_model2 = fe_model2.align()
        #
        # ef_model2 = IBMModel2(ef_bitext, ef_model1.theta, tgt_vocab, src_vocab, max_iter=10)
        # ef_model2.train()
        # ef_alignments_model2 = ef_model2.align()

        fe_alignments, ef_alignments = ef_alignments_model2, ef_alignments_model2

        with open(sys.argv[3] + '.f2e.model1', 'w') as f:
            for cur_alignments in fe_alignments_model1:
                line = ' '.join('%d-%d' % (i, j) for j, i in cur_alignments)
                f.write(line + '\n')

        with open(sys.argv[3] + '.e2f.model1', 'w') as f:
            for cur_alignments in ef_alignments_model1:
                line = ' '.join('%d-%d' % (i, j) for j, i in cur_alignments)
                f.write(line + '\n')

        with open(sys.argv[3] + '.f2e.model2', 'w') as f:
            for cur_alignments in fe_alignments_model2:
                line = ' '.join('%d-%d' % (i, j) for j, i in cur_alignments)
                f.write(line + '\n')

        with open(sys.argv[3] + '.e2f.model2', 'w') as f:
            for cur_alignments in ef_alignments_model2:
                line = ' '.join('%d-%d' % (i, j) for j, i in cur_alignments)
                f.write(line + '\n')
    elif LOAD_PARAMS:
        print('load models ...')

        fe_model2_theta, fe_model2_align_prob = deserialize_from_file('output_no_null_efhack/model2.f2e.bin')

        fe_model2 = IBMModel2(fe_bitext, fe_model2_theta, src_vocab, tgt_vocab, max_iter=10, dir='f2e')
        fe_model2.align_prob = fe_model2_align_prob
        fe_alignments_model2 = fe_model2.align()

        ef_model2_theta, ef_model2_align_prob = deserialize_from_file('output_no_null_efhack/model2.e2f.bin')

        ef_model2 = IBMModel2(ef_bitext, ef_model2_theta, tgt_vocab, src_vocab, max_iter=10, dir='e2f')
        ef_model2.align_prob = ef_model2_align_prob
        ef_alignments_model2 = ef_model2.align()

        fe_alignments, ef_alignments = fe_alignments_model2, ef_alignments_model2
    else:
        print('read in pre-trained alignments... [%s] and [%s]' % (sys.argv[4], sys.argv[5]))

        # f2e alignments
        fe_alignments = []
        for line in open(sys.argv[4]):
            d = line.strip().split(' ')
            cur_alignments = []
            for e in d:
                e = e.split('-')
                cur_alignments.append((int(e[1]), int(e[0])))

            fe_alignments.append(cur_alignments)

        # e2f alignments
        ef_alignments = []
        for line in open(sys.argv[5]):
            d = line.strip().split(' ')
            cur_alignments = []
            for e in d:
                e = e.split('-')
                cur_alignments.append((int(e[1]), int(e[0])))

            ef_alignments.append(cur_alignments)

    with open(sys.argv[3], 'w') as f:
        for idx, (f2e_cur_alignments, e2f_cur_alignments) in enumerate(zip(fe_alignments, ef_alignments)):
            # valid_alignments = [(j, i) for j, i in fe_cur_alignments if (i, j) in ef_cur_alignments]  # intersection
            # valid_alignments = sorted(set(fe_cur_alignments).union(set((j, i) for i, j in ef_cur_alignments)))
            # f2e_align_str = ' '.join('%d-%d' % (j, i) for j, i in f2e_cur_alignments)
            # e2f_align_str = ' '.join('%d-%d' % (j, i) for i, j in e2f_cur_alignments)

            src_sent = bitext[idx][0]
            tgt_sent = bitext[idx][1]

            # e2f_cur_alignments = [(j, i) for i, j in e2f_cur_alignments]
            # alignments = e2f_cur_alignments

            alignments = set(f2e_cur_alignments).union(set(e2f_cur_alignments))

            alignments = grow_diag_final_and(len(src_sent), len(tgt_sent), e2f_cur_alignments, f2e_cur_alignments)
            # print(alignments)
            # if len(alignments) == 0:
            #    alignments = e2f_cur_alignments

            line = ' '.join('%d-%d' % (i, j) for j, i in alignments)
            # line = ' '.join('%d-%d' % (i, j) for j, i in e2f_cur_alignments)
            f.write(line + '\n')

