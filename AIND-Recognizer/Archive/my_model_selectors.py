import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        num_of_components = range(self.min_n_components, self.max_n_components + 1)
        bic_scores = []
        try:
            for num_states in num_of_components:
                model = self.base_model(num_states)
                log_likelihood = model.score(self.X, self.lengths)
                num_data_points = self.X.shape[0]
                num_free_params = (num_states ** 2) + (2 * num_states * num_data_points) - 1
                bic_score = (-2 * log_likelihood) + num_free_params * math.log(num_data_points)
                bic_scores.append(bic_score)
        except:
             pass
        num_states = num_of_components[np.argmax(bic_scores)] if bic_scores else self.n_constant
        return  self.base_model(num_states)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        num_of_components = range(self.min_n_components, self.max_n_components + 1)
        dic_scores = []
        logs_likelihood = []
        try:
            for num_states in num_of_components:
                model = self.base_model(num_states)
                log_likelihood = model.score(self.X, self.lengths)
                logs_likelihood.append(log_likelihood)
            for log_likelihood in logs_likelihood:
                dic_score = log_likelihood - (sum(logs_likelihood) -log_likelihood)/(len(num_of_components)-1)
                dic_scores.append(dic_score)
        except:
            pass

        num_states = num_of_components[np.argmax(dic_scores)] if dic_scores else self.n_constant
        return self.base_model(num_states)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def train_cv_model(self, num_states, X_train, length_train):
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, length_train)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        num_of_components = range(self.min_n_components, self.max_n_components + 1)
        split_method = KFold(n_splits=3)
        try:
            mean_logs_likelihood = []
            for num_states in num_of_components:
                k_fold_scores = []
                for train_idx, test_idx in split_method.split(self.sequences):
                    X_train, length_train = combine_sequences(train_idx, self.sequences)
                    X_test, length_test = combine_sequences(test_idx, self.sequences)
                    model = self.train_cv_model(num_states, X_train, length_train)
                    k_fold_scores.append(model.score(X_test, length_test))
                mean_logs_likelihood.append(np.mean(k_fold_scores))

        except:
            pass
        states = num_of_components[np.argmax(mean_logs_likelihood)] if mean_logs_likelihood else self.n_constant
        return self.base_model(states)



