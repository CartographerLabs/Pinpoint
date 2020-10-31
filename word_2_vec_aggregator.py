from gensim.models import Word2Vec

class word_2_vec_aggregator():

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

        print(list_of_sentences_in_nested_list)

        model = Word2Vec(min_count=1)
        model.build_vocab(list_of_sentences_in_nested_list)  # prepare the model vocabulary
        model.train(list_of_sentences_in_nested_list, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors

        return model