import numpy as np

from constants import INVERT, ADD_NOISE_TO_DATA, NOISE_PROBS
from data_augmentation import add_noise_to_data
from data_gen import LazyDataLoader
from utils import get_chars_and_ctable, get_TOKEN_INDICES

print('Vectorization...')

DATA_LOADER = LazyDataLoader()

INPUT_MAX_LEN, OUTPUT_MAX_LEN, TRAINING_SIZE = DATA_LOADER.statistics()

TOKEN_INDICES = get_TOKEN_INDICES()

chars, c_table = get_chars_and_ctable()

questions = []
expected = []
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    x, y = DATA_LOADER.next()
    # Pad the data with spaces such that it is always MAXLEN.
    q = x
    query = q
    ans = y

    if ADD_NOISE_TO_DATA:
        #print('Old query =', query, end='  |   ')
        query, _ = add_noise_to_data(input_str=query, probs=NOISE_PROBS, vocabulary=chars)
        #print('Query =', query, '  |   Noise type =', noise_type)

    if INVERT:
        # 反转输入序列
        query = query[::-1]

    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

x = np.zeros((len(questions), INPUT_MAX_LEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), OUTPUT_MAX_LEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):

    x[i] = c_table.encode(sentence, INPUT_MAX_LEN)
for i, sentence in enumerate(expected):

    y[i] = c_table.encode(sentence, OUTPUT_MAX_LEN)


indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

np.savez_compressed('x_y.npz', x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)

print('Done... File is x_y.npz')
