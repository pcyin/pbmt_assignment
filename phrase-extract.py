# -*- encoding: utf-8 -*-

from __future__ import print_function

from itertools import chain

import numpy as np
import math
import sys
from collections import defaultdict, Counter


def quasi_consec(tp, f_aligned_words):
    """
    tp must be an ordered list
    """
    j1 = tp[0]
    j2 = tp[-1]
    is_consec = len(tp) == j2 - j1 + 1

    if is_consec:
        return True

    tp_ptr = 0
    for j in xrange(j1, j2 + 1):
        if j == tp[tp_ptr]:
            tp_ptr += 1
        elif len(f_aligned_words[j]) > 0:
            return False

    return True


class PhraseExtractor(object):
    def __init__(self, bitext, alignments, max_phrase_len=3):
        self.bitext = bitext
        self.alignments = alignments
        self.max_phrase_len = max_phrase_len

    def extract_phrase_in_sent(self, e_aligned_words, f_aligned_words, e, f):
        extracted_phrases = []
        # Loop over all sub-strings in the E
        for i1 in xrange(len(e)):
            for i2 in xrange(i1, min(len(e), i1 + self.max_phrase_len)):
                # Get all positions in F that correspond to the substring from i1 to i2 in E (inclusive)
                tp = sorted(set(chain(*[e_aligned_words[i] for i in xrange(i1, i2 + 1)])))
                if len(tp) != 0 and quasi_consec(tp, f_aligned_words):
                    j1 = tp[0]
                    j2 = tp[-1]

                    # Get all positions in E that correspond to the substring from j1 to j2 in F (inclusive)
                    sp = [i for i in xrange(len(e)) if any(j1 <= j <= j2 for j in e_aligned_words[i])]

                    # Check that all elements in sp fall between i1 and i2 (inclusive)
                    if len(sp) != 0 and all(i1 <= i <= i2 for i in sp):
                        e_phrase = e[i1: i2 + 1]
                        f_phrase = f[j1: j2 + 1]
                        # extracted_phrases.append((e_phrase, f_phrase))

                        # Extend source phrase by adding unaligned words
                        j1_prime = j1
                        while j1_prime >= 0 and j1_prime == j1 or len(f_aligned_words[j1]) == 0:  # Check that j1 is unaligned
                            j2_prime = j2
                            while j2_prime < len(f) and j2_prime == j2 or len(f_aligned_words[j2]) == 0:  # Check that j2 is unaligned
                                f_phrase = f[j1_prime: j2_prime + 1]
                                extracted_phrases.append((' '.join(f_phrase), ' '.join(e_phrase)))
                                j2_prime += 1

                            j1_prime -= 1

        return extracted_phrases

    def extract_phrase(self):
        extracted_phrases_counts = defaultdict(lambda: defaultdict(float))
        for (f, e), cur_alignments in zip(self.bitext, self.alignments):
            f_aligned_words = defaultdict(set)
            e_aligned_words = defaultdict(set)
            for j, i in cur_alignments:
                f_aligned_words[j].add(i)
                e_aligned_words[i].add(j)

            cur_extracted_phrases = self.extract_phrase_in_sent(e_aligned_words, f_aligned_words, e, f)
            for f_phrase, e_phrase in cur_extracted_phrases:
                extracted_phrases_counts[e_phrase][f_phrase] += 1.
            #     print(f_phrase + ' --- ' + e_phrase)


        # compute p(f|e)
        for e_phrase in extracted_phrases_counts:
            e_total = sum(extracted_phrases_counts[e_phrase].values())
            for f_phrase in extracted_phrases_counts[e_phrase]:
                extracted_phrases_counts[e_phrase][f_phrase] /= e_total

        return extracted_phrases_counts

if __name__ == '__main__':
    bitext = []
    for src_sent, tgt_sent in zip(open(sys.argv[1]), open(sys.argv[2])):
        src_words = src_sent.strip().split(' ')
        tgt_words = tgt_sent.strip().split(' ')

        bitext.append((src_words, tgt_words))

    # read in alignments
    alignments = []
    for line in open(sys.argv[3]):
        # source, target
        data = line.strip().split(' ')
        cur_alignments = []
        for entry in data:
            d = entry.split('-')
            j = int(d[1])
            i = int(d[0])

            cur_alignments.append((j, i))

        alignments.append(cur_alignments)

    extractor = PhraseExtractor(bitext, alignments)
    extracted_phrases_counts = extractor.extract_phrase()

    with open(sys.argv[4], 'w') as f:
        for e_phrase in extracted_phrases_counts:
            for f_phrase, p_f_given_e in extracted_phrases_counts[e_phrase].iteritems():
                score = -math.log(p_f_given_e)
                if score == 0.: score = 0.
                f.write('%s\t%s\t%.4f\n' % (f_phrase, e_phrase, score))
