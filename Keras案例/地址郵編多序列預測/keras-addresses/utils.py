import pickle

import numpy as np

from constants import ADDRESS_FILE


def get_INDICES_TOKEN():
    return pickle.load(open('indices_token.pkl', 'rb'))


def get_TOKEN_INDICES():
    return pickle.load(open('token_indices.pkl', 'rb'))


def get_VOCAB_SIZE():
    return len(get_TOKEN_INDICES())


class CharacterTable(object):
    """
        编码和解码
    """

    def __init__(self, chars):
        """
         chars: 出现的不同字符数量.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """
        One hot encode.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)





def get_chars_and_ctable():
    chars = ''.join(list(get_TOKEN_INDICES().values()))
    ctable = CharacterTable(chars)
    return chars, ctable


def build_vocabulary():
    vocabulary = set()
    with open(ADDRESS_FILE, 'rb') as r:
        for l in r.readlines()[:100000]:
            y, x = l.decode('utf8').strip().split('　')
            for element in list(y):
                vocabulary.add(element)
            for element in list(x):
                vocabulary.add(element)
    vocabulary = sorted(list(vocabulary))
    print(vocabulary)
    token_indices = dict((c, i) for (c, i) in enumerate(vocabulary))
    indices_token = dict((i, c) for (c, i) in enumerate(vocabulary))

    with open('token_indices.pkl', 'wb') as w:
        pickle.dump(obj=token_indices, file=w)

    with open('indices_token.pkl', 'wb') as w:
        pickle.dump(obj=indices_token, file=w)

    print('Done... File is token_indices.pkl')
    print('Done... File is indices_token.pkl')


if __name__ == '__main__':
    build_vocabulary()
