import numpy as np


def _remove_long_seq(maxlen, seq, label):
    """
    Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.

    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


def load_data_fashion_mnist(dirname='datasets'):
    """
    Loads the Fashion-MNIST dataset.

    Returns:
       Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    See: tensorflow.keras.datasets.fashion_mnist.load_data()
    """
    import os
    import gzip

    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(os.path.dirname(__file__), dirname, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def load_data_imdb(path='imdb.npz',
                   dirname='datasets',
                   num_words=None,
                   skip_top=0,
                   maxlen=None,
                   seed=113,
                   start_char=1,
                   oov_char=2,
                   index_from=3,
                   **kwargs):
    """
    Loads the IMDB dataset.

    Arguments:
        path: path where is the dataset
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        maxlen: sequences longer than this will be filtered out.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.
        **kwargs: Used for backwards compatibility.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Raises:
        ValueError: in case `maxlen` is so low
            that no input sequence could be kept.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.

    See: tensorflow.keras.datasets.imdb.load_data()
    """
    import os

    # Legacy support
    if 'nb_words' in kwargs:
        print('The `nb_words` argument in `load_data` has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    with np.load(os.path.join(os.path.dirname(__file__), dirname, path)) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path='imdb_word_index.json', dirname='datasets'):
    """
    Retrieves the dictionary mapping word indices back to words.

    Arguments:
        path: path where is the dataset

    Returns:
        The word index dictionary.

    See: tensorflow.keras.datasets.imdb.get_word_index()
    """
    import os
    import json

    with open(os.path.join(os.path.dirname(__file__), dirname, path)) as f:
        return json.load(f)


def load_data_boston_housing(path='boston_housing.npz', dirname='datasets', test_split=0.2, seed=113):
    """
    Loads the Boston Housing dataset.

    Arguments:
        path: path where is the dataset
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data before computing the test split.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    See: tensorflow.keras.datasets.boston_housing.load_data()
    """
    import os

    assert 0 <= test_split < 1
    with np.load(os.path.join(os.path.dirname(__file__), dirname, path)) as f:
        x = f['x']
        y = f['y']

    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


def load_data_mnist(path='mnist.npz', dirname='datasets'):
    """
    Loads the MNIST dataset.

    Arguments:
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    See: tensorflow.keras.datasets.mnist.load_data()
    """
    import os

    with np.load(os.path.join(os.path.dirname(__file__), dirname, path)) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data_mnist()
    print(len(x_train), len(x_test))
