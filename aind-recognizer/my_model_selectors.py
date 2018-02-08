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
        
        lowest_bic = float('inf')
        current_bic = float('inf')
        best_model = None

        for comp in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(comp)
                log_lhd = hmm_model.score(self.X, self.lengths)
                param = (comp ** 2) + (2 * hmm_model.n_features * comp) - 1
                current_bic = (-2 * log_lhd) + (param * np.log(hmm_model.n_features))
            except Exception as e:
                pass 

            if current_bic < lowest_bic:
                lowest_bic = current_bic
                best_model = hmm_model
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calc_log_likelihood_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]
        
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        other_words = []
        all_models = []
        all_dics = []
        highest_dic = None

        for word in self.words:
            if word is not self.this_word:
                other_words.append(self.hwords[word])
        try:
            for state in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(state)
                log_lhd = hmm_model.score(self.X, self.lengths)
                all_models.append((log_lhd, hmm_model))
        except Exception as e:
            pass         

        for _, model in enumerate(all_models):
            log_lhd, hmm_model = model
            current_dic = log_lhd - np.mean(self.calc_log_likelihood_other_words(model, other_words))
            all_dics.append((current_dic, model[1]))

        if all_dics:
            return max(all_dics, key = lambda x: x[0])[1]
        else:
            return None 


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        log_lhd = []
        best_score_cv = None
        avg_score_cv = None
        best_model = None

        for state in range(self.min_n_components, self.max_n_components + 1):
            try:    
                if len(self.sequences) > 2:
                    split_method = KFold();
                    for train_index, test_index in split_method.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train_index, self.sequences) 
                        X_test, length_test = combine_sequences(test_index, self.sequences) 
                        hmm_model = self.base_model(state)
                        log_lhd_current = hmm_model.score(X_test, length_test)
                else:
                    hmm_model = self.base_model(state)
                    log_lhd_current = hmm_model.score(self.X, self.lengths)
                log_lhd.append(log_lhd_current)
                avg_score_cv = np.mean(log_lhd)

                if best_score_cv is not None and best_score_cv < avg_score_cv:
                    best_score_cv = avg_score_cv
                    best_model = hmm_model
                if best_model is None:
                    best_model = hmm_model
            except Exception as e:
                pass
        return best_model
