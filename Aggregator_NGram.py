from sklearn.feature_extraction.text import CountVectorizer

from Logger import *

c_vec = CountVectorizer(ngram_range=(1, 5))

class n_gram_aggregator():
    """
    This class is used to retrieve the most common NGrams for a given dataset corpus.
    """

    def _get_average_ngram_count(self, n_grams_dict):
        """
        takes a dict of Ngrams and identifies the average weighting
        :param n_grams_dict:
        :return:
        """
        all_count = []
        for n_gram in n_grams_dict:
            ng_count = n_grams_dict[n_gram]
            all_count.append(ng_count)

        average_count = sum(all_count) / len(all_count)
        #print(all_count)
        return average_count

    def _get_all_ngrams(self, data):
        """
        Returns all ngrams (tri, bi, and uni) for a given piece of text
        :param data:
        :return:
        """

        if type(data) is not list:
            data = [data]

        # input to fit_transform() should be an iterable with strings
        ngrams = c_vec.fit_transform(data)

        # needs to happen after fit_transform()
        vocab = c_vec.vocabulary_

        count_values = ngrams.toarray().sum(axis=0)

        # output n-grams
        uni_grams = {}
        bi_grams = {}
        tri_grams = {}

        for ng_count, ng_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
            sentence_length = len(ng_text.split(" "))

            if sentence_length == 3:
                tri_grams[ng_text] = ng_count
            elif sentence_length == 2:
                bi_grams[ng_text] = ng_count
            elif sentence_length == 1:
                uni_grams[ng_text] = ng_count

        return uni_grams,bi_grams,tri_grams

    def _get_popular_ngrams(self, ngrams_dict):
        """
        Returns ngrams for a given piece of text that are the most popular (i.e. their weighting is
        above the average ngram wighting)
        :param ngrams_dict:
        :return:
        """
        average_count = self._get_average_ngram_count(ngrams_dict)

        popular_ngrams = {}
        for n_gram in ngrams_dict:
            ng_count = ngrams_dict[n_gram]

            if ng_count >= average_count:
                popular_ngrams[n_gram] = ng_count
        return popular_ngrams

    def get_ngrams(self, data = None, file_name_to_read = None):
        """
        Wrapper function for returning uni, bi, and tri grams that are the most popular (above the average weighting in
        a given piece of text).
        :param data:
        :param file_name_to_read:
        :return:
        """
        logger().print_message("Getting Ngrams")

        if data is None and file_name_to_read is None:
            raise Exception("No data supplied to retrieve n_grams")

        if data is None and file_name_to_read is not  None:
            with open(file_name_to_read, 'r') as file_to_read:
                data = file_to_read.read()

        uni_grams,bi_grams,tri_grams = self._get_all_ngrams(data)

        popular_uni_grams = list(self._get_popular_ngrams(uni_grams).keys())
        popular_bi_grams = list(self._get_popular_ngrams(bi_grams).keys())
        popular_tri_grams = list(self._get_popular_ngrams(tri_grams).keys())

        return popular_uni_grams,popular_bi_grams,popular_tri_grams
