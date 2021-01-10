from gensim.models import Word2Vec


class word_2_vec_aggregator():
    """
    A wrapper function around gensim used for creating a word 2 vec model
    """

    def get_model(self, list_of_sentences):
        """
        Used to retrieve the model
        :param list_of_sentences:
        :return: the model
        """

        list_of_sentences_in_nested_list = []

        for sentence in list_of_sentences:

            # Skip unigrams
            if " " not in sentence:
                continue

            list_of_sentences_in_nested_list.append(sentence.split(" "))

        model = Word2Vec(min_count=1, window=5)  # vector size of 100 and window size of 5?
        model.build_vocab(list_of_sentences_in_nested_list)  # prepare the model vocabulary
        model.train(list_of_sentences_in_nested_list, total_examples=model.corpus_count,
                    epochs=model.iter)  # train word vectors

        return model
