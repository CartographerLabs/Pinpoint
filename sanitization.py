from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from nltk import *
import os.path

#nltk.download() #todo how to get this to run once?

class sanitization():

    def sanitize(self, text, force_new_data_and_dont_persisit = False):
        """
        Entry function for sanitizing text
        :param text:
        :param force_new_data_and_dont_persisit:
        :return: sanitized text
        """
        sanitize_file_name = "sanitized_text.txt"
        final_text = ""

        # If a file exists don't sanitize given text
        if os.path.isfile(sanitize_file_name) and not force_new_data_and_dont_persisit:
            print("Sanitized file exists. Using data")

            with open(sanitize_file_name, 'r', encoding="utf8") as file_to_write:
                final_text = file_to_write.read()

        else:
            total_words = len(text.split(" "))
            number = 0
            print("Starting sanitization... {} words to go".format(total_words))
            for word in text.split(" "):
                number = number + 1
                word = self.remove_non_alpha(word)
                word = self.lower(word)
                word = self.stemmer(word)
                word = self.remove_stop_words(word)
                word = self.remove_small_words(word)

                if word is None:
                    continue

                final_text = final_text + word + " "
                print("Completed {} of {} sanitized words".format(number, total_words))

            final_text = final_text.replace("  "," ")

            if not force_new_data_and_dont_persisit:
                with open(sanitize_file_name, 'w', encoding="utf8") as file_to_write:
                    file_to_write.write(final_text)

        return final_text

    def stemmer(self, word):
        """
        Get stemms of words
        :param word:
        :return:
        """

        porter = PorterStemmer()
        #lancaster = LancasterStemmer()
        #stemmed_word = lancaster.stem(word)
        stemmed_word = porter.stem(word)

        return stemmed_word

    def lower(self, word):
        """
        get the lower case representation of words
        :param word:
        :return:
        """
        return word.lower()

    def remove_stop_words(self, text):
        """
        Remove stop words
        :param text:
        :return:
        """

        text_without_stopwords = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]

        final_string = ""

        for word in text_without_stopwords:
            final_string = final_string+ word+ " "

        return final_string

    def remove_non_alpha(self, word):
        """
        Removes non alphabet characters (Excluding spaces)
        :param word:
        :return:
        """
        word = word.replace("\n"," ").replace("\t", " ").replace("  ", " ")
        regex = re.compile('[^a-zA-Z ]')
        # First parameter is the replacement, second parameter is your input string
        return regex.sub('', word)
        # Out: 'abdE'

    def remove_small_words(self, word, length_to_remove_if_not_equal = 4):
        """
        Removes words that are too small, defaults to words words length 3 characters or below which are removed.
        :param word:
        :param length_to_remove_if_not_equal:
        :return: "" if word below 3 characters or the word
        """

        new_word = ""
        if len(word) >= length_to_remove_if_not_equal:
            new_word = word

        return new_word
