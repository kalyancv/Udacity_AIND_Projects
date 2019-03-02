import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    X_lengths = test_set.get_all_Xlengths()
    for X, lengths in X_lengths.values():
        word_logs_l = {}
        for word, model in models.items():
            try:
                log_likelihood = model.score(X, lengths)
                word_logs_l[word] = log_likelihood
            except:
                word_logs_l[word] = float("-inf")
        probabilities.append(word_logs_l)
        guesses.append(max(word_logs_l, key = word_logs_l.get))
    return probabilities, guesses