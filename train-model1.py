# -*- encoding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict


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
    def __init__(self, bitext, src_vocab, tgt_vocab, max_iter=10):
        self.bitext = bitext
        self.max_iter = max_iter
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.epsilon = 1. / max(len(src_sent) for src_sent, tgt_sent in bitext)
        self.theta = None

    def train(self):
        print('start training... Source Vocab[%d], Target Vocab[%d]' % (len(self.src_vocab), len(self.tgt_vocab)))

        # initialize parameters
        p_f_e_init = 1. / len(self.src_vocab)
        theta = defaultdict(lambda: p_f_e_init)

        def _time_prior(p, e_i, e_len):
            # if e_i == self.tgt_vocab['<null>']:
            #     return p * 0.2
            # return p * 0.8 / (e_len - 1)
            return p

        for t in xrange(self.max_iter):
            # E step
            counts_e = defaultdict(float)
            counts = defaultdict(float)

            for src_sent, tgt_sent in self.bitext:
                src_sent = [self.src_vocab[w] for w in src_sent]
                tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]
                for j in xrange(len(src_sent)):
                    f_j = src_sent[j]
                    marginal = sum(_time_prior(theta[(e_i, f_j)], e_i, len(tgt_sent)) for e_i in tgt_sent)

                    for i in xrange(len(tgt_sent)):
                        e_i = tgt_sent[i]
                        align_posterior = theta[(e_i, f_j)] / marginal
                        counts[(e_i, f_j)] += align_posterior
                        counts_e[e_i] += align_posterior

            # M step
            for e_i, f_j in counts:
                theta[(e_i, f_j)] = counts[(e_i, f_j)] / counts_e[e_i]

            # compute log-likelihood
            ll = 0.
            for src_sent, tgt_sent in self.bitext:
                src_sent = [self.src_vocab[w] for w in src_sent]
                tgt_sent = [self.tgt_vocab[w] for w in tgt_sent]

                p_F_given_E = np.log(self.epsilon) - len(src_sent) * np.log(len(tgt_sent))
                for f_j in src_sent:
                    p_fj = 0.
                    for e_i in tgt_sent:
                        p_fj += theta[(e_i, f_j)]

                    p_F_given_E += np.log(p_fj)

                ll += p_F_given_E / len(src_sent)

            ll /= len(self.bitext)
            print('avg. ll per word: %f' % ll)

            self.theta = theta

    def align(self):
        alignments = []
        for idx, (f, e) in enumerate(self.bitext):
            cur_alignments = []
            for j in xrange(len(f)):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)

                f_j = f[j]

                max_prob = -9999.
                max_i = -1
                for i, e_i in enumerate(e):
                    prob = self.theta[(self.tgt_vocab[e_i], self.src_vocab[f_j])]
                    if prob > max_prob:
                        max_prob = prob
                        max_i = i

                if max_i == len(e) - 1:
                    continue # skip the last null word in E

                cur_alignments.append((j, max_i))

            alignments.append(cur_alignments)

        return alignments

if __name__ == '__main__':
    src_vocab = defaultdict(lambda: len(src_vocab))
    tgt_vocab = defaultdict(lambda: len(tgt_vocab))
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

    fe_bitext = [(src_sent, tgt_sent + ['<null>']) for src_sent, tgt_sent in bitext]
    ef_bitext = [(tgt_sent, src_sent + ['<null>']) for src_sent, tgt_sent in bitext]

    # fe_model1 = IBMModel1(fe_bitext, src_vocab, tgt_vocab)
    # fe_model1.train()
    # fe_alignments = fe_model1.align()

    ef_model1 = IBMModel1(ef_bitext, tgt_vocab, src_vocab)
    ef_model1.train()
    ef_alignments = ef_model1.align()

    # with open(sys.argv[3] + '.fe', 'w') as f:
    #     for cur_alignments in fe_alignments:
    #         line = ' '.join('%d-%d' % (i, j) for j, i in cur_alignments)
    #         f.write(line + '\n')

    with open(sys.argv[3], 'w') as f:
        for cur_alignments in ef_alignments:
            line = ' '.join('%d-%d' % (i, j) for i, j in cur_alignments)
            f.write(line + '\n')

    # with open(sys.argv[3], 'w') as f:
    #     for fe_cur_alignments, ef_cur_alignments in zip(fe_alignments, ef_alignments):
    #         valid_alignments = [(j, i) for j, i in fe_cur_alignments if (i, j) in ef_cur_alignments]
    #         line = ' '.join('%d-%d' % (i, j) for j, i in valid_alignments)
    #         # line = ' '.join('%d-%d' % (i, j) for i, j in ef_cur_alignments)
    #         f.write(line + '\n')
