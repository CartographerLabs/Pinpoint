from n_gram_aggregator import *
from tf_idf_aggregator import *
from sanitization import *
from word_2_vec_aggregator import *

# Read dataset
with open("data-set.txt", 'r', encoding="utf8") as file_to_read:
    data_to_analyse = file_to_read.read()

# clean data set
clean_data = sanitization().sanitize(data_to_analyse)

# get ngrams
uni_grams,bi_grams,tri_grams = ngram_aggregator = n_gram_aggregator().get_ngrams(clean_data)
ngrams = uni_grams + bi_grams + tri_grams

# get tf idf scores
tf_idf_scores = tf_idf_aggregator().get_tf_idf_scores(bi_grams, clean_data)

# create a word 2 vec model
model = word_2_vec_aggregator().get_model(list_of_sentences=ngrams)
predicted_word = model.predict_output_word(["psycholog"])
print(predicted_word)