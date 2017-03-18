# -*- encoding: utf-8 -*-

from __future__ import print_function

from itertools import chain

import numpy as np
import math
import sys
from collections import defaultdict, Counter


def create_fst(phrases):
    f_fst = open(sys.argv[2], 'w')

    prefix_node_map = dict()
    prefix_node_map[''] = 0
    for f_phrase, e_phrase, score in phrases:
        f_words = f_phrase.split(' ')
        e_words = e_phrase.split(' ')

        f_phrase_len = len(f_words)
        e_phrase_len = len(e_words)

        prev_node_prefix = ''
        for j in xrange(f_phrase_len):
            f_j = f_words[j]
            if j == 0:
                prefix = f_words[j]
            else:
                prefix = prev_node_prefix + ' ' + f_words[j]

            if prefix not in prefix_node_map:
                node_id = len(prefix_node_map)
                prefix_node_map[prefix] = node_id
                prev_node_id = prefix_node_map[prev_node_prefix]

                f_fst.write('%d %d %s <eps>\n' % (prev_node_id, node_id, f_j))

            prev_node_prefix = prefix

        for i in xrange(e_phrase_len):
            e_i = e_words[i]
            if i == 0:
                prefix = prev_node_prefix + ' ||| ' + e_words[i]
            else:
                prefix = prev_node_prefix + ' ' + e_words[i]

            if prefix not in prefix_node_map:
                node_id = len(prefix_node_map)
                prefix_node_map[prefix] = node_id
                prev_node_id = prefix_node_map[prev_node_prefix]

                f_fst.write('%d %d <eps> %s\n' % (prev_node_id, node_id, e_i))

            prev_node_prefix = prefix

        f_fst.write('%d 0 <eps> <eps> %.4f\n' % (prefix_node_map[prev_node_prefix], score))

    f_fst.write('0 0 </s> </s>\n')
    f_fst.write('0 0 <unk> <unk>\n')
    f_fst.write('0\n')
    f_fst.close()


if __name__ == '__main__':
    phrases = []
    for line in open(sys.argv[1]):
        data = line.strip().split('\t')
        phrases.append((data[0], data[1], float(data[2])))

    create_fst(phrases)