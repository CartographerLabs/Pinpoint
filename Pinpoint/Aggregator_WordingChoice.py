import os


class wording_choice_aggregator():
    """
    A class used for retrieving frequencies based on wording in a message
    """

    def get_frequency_of_capatalised_words(self, text):
        """
        A function used to retrieve the frequencies of capitalised words in a dataset
        :param text:
        :return: the frequency of capitalised words in a dataset
        """
        number_of_capatalised_words = 0
        for word in text.split(" "):
            if word.isupper():
                number_of_capatalised_words = number_of_capatalised_words + 1

        total_number_of_words = len(text.split(" "))
        frequency = number_of_capatalised_words / total_number_of_words

        return frequency

    def get_frequency_of_violent_or_curse_words(self, text):
        """
        A function ued for retrieving the frequencies of violent words in a dataset
        :param text:
        :return: the frequency of violent words in a dataset
        """

        dataset_folder = os.path.join(os.getcwd(), "violent_or_curse_word_datasets")

        list_of_violent_or_curse_words = []

        # Retrieves all words in all of the files in the violent or curse word datasets
        for filename in os.listdir(dataset_folder):
            with open(os.path.join(dataset_folder, filename), 'r') as file:

                for line in file.readlines():
                    line = line.strip().replace("\n", " ").replace(",","")
                    list_of_violent_or_curse_words.append(line)

        number_of_swear_words = 0
        for word in text.split(" "):
            if word in list_of_violent_or_curse_words:
                number_of_swear_words = number_of_swear_words + 1

        total_number_of_words = len(text.split(" "))
        frequency = number_of_swear_words / total_number_of_words
        return frequency
