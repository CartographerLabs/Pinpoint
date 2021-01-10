from sklearn.feature_extraction.text import TfidfVectorizer

from Pinpoint.Logger import *


class tf_idf_aggregator():
    """
    A wrapper class around SKlearn for retrieving TF-IDF scores.
    """

    def get_tf_idf_scores(self, ngrams_vocabulary, corpus_data=None, file_name_to_read=None):
        """
        Used to generate a TF IDF score based of a vocabulary of Ngrams and a data corpus.
        :param ngrams_vocabulary:
        :param corpus_data:
        :param file_name_to_read:
        :return: a dictionary of the pairing name and their score
        """
        logger.print_message("Getting TF IDF scores")

        if corpus_data is None and file_name_to_read is None:
            raise Exception("No data supplied to retrieve n_grams")

        if corpus_data is None and file_name_to_read is not None:
            with open(file_name_to_read, 'r') as file_to_read:
                corpus_data = file_to_read.read()

        tfidf = TfidfVectorizer(vocabulary=ngrams_vocabulary, stop_words='english', ngram_range=(1, 2))
        tfs = tfidf.fit_transform([corpus_data])

        feature_names = tfidf.get_feature_names()
        corpus_index = [n for n in corpus_data]
        rows, cols = tfs.nonzero()

        dict_of_scores = {}

        for row, col in zip(rows, cols):
            dict_of_scores[feature_names[col]] = tfs[row, col]
            logger.print_message((feature_names[col], corpus_index[row]), tfs[row, col])

        return dict_of_scores
