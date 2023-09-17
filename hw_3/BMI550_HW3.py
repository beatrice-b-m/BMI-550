import nltk
import numpy as np
from dataclasses import dataclass
from operator import attrgetter
from itertools import chain
from collections import OrderedDict

"""
1. 
Compute the jaccard similarities between scientificpub1 and
scientificpub2, and between scientificpub2 and scientificpub3. Which
pair has higher jaccard similarity?

2. 
    a)  Generate the document vectors for the scientific texts using one-
        hot representations.
    b)  Compute the cosine similarity between scientificpub1 and
        scientificpub2.
"""

def load_txt_file(file_path):
    """
    load a text file from a path
    """
    with open(file_path, 'r') as file:
        txt_file = file.read()
    return txt_file

def preprocess_tokens(text: str, punctuation_str: str = "][)(,'‘’.1234567890:;%“”",
                    replace_list: list = ['-', '/']):
    """
    preprocess text to lowercase it, remove punctuation, tokenize it, remove
    stopwords, and stem the words. returns a list of preprocessed tokens
    im reusing this function from hw2 :)
    """
    # lowercase text then strip punctuation and non-informative
    # characters from the it
    stripped_text = strip_punctuation(
        text.lower(),
        punctuation_str=punctuation_str,
        replace_list=replace_list
    )

    # tokenize text
    text_tokens = nltk.tokenize.word_tokenize(stripped_text)

    # eliminate stopwords
    english_stopwords = nltk.corpus.stopwords.words('english')
    cleaned_tokens = [t for t in text_tokens if t not in english_stopwords]

    # stem words
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(t) for t in cleaned_tokens]

    return stemmed_tokens

def strip_punctuation(target_str: str, punctuation_str: str = "][)(,",
                      replace_list: list = []):
    """
    strip punctuation in punctuation_str from target_str and replace
    elements in replace_list with ' '
    i also reused this function from hw2 :)
    """
    out_str = target_str.translate({ord(c): None for c in punctuation_str})
    for s in ["\n"] + replace_list:
        out_str = out_str.replace(s, " ")
    return out_str

def jaccard_similarity(tokens_a, tokens_b):
    """
    jaccard similarity is also known as intersection over union
    and is calculated by dividing the intersection of two sets
    by their union
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    iou = len(intersection) / len(union)
    return iou

@dataclass
class DocComp:
    label_a: str
    label_b: str
    jaccard_similarity: float

print("\nQuestion 1", "-"*70)
print("""Compute the jaccard similarities between scientificpub1 and
scientificpub2, and between scientificpub2 and scientificpub3. Which
pair has higher jaccard similarity?""")

# load the documents
file_path_list = ['./hw_3/content/scientificpub1', './hw_3/content/scientificpub2', './hw_3/content/scientificpub3']
file_list = [load_txt_file(file_path) for file_path in file_path_list]

# get tokens for the documents
file_token_list = [preprocess_tokens(text_file) for text_file in file_list]

# get document labels and zip with the token lists
doc_label_list = ['scientificpub1', 'scientificpub2', 'scientificpub3']
zipped_token_list = list(zip(doc_label_list, file_token_list))

jaccard_list = []

# iterate over the range of document combinations
for i in range(len(doc_label_list) - 1):
    # unpack labels and tokens
    label_a, token_list_a = zipped_token_list[i]
    label_b, token_list_b = zipped_token_list[i + 1]

    # store the jaccard similarity in a DocComp dataclass object
    document_comparison = DocComp(
        label_a, 
        label_b, 
        jaccard_similarity(token_list_a, token_list_b)
    )

    # append the DocComp object to jaccard_list
    jaccard_list.append(document_comparison)

# iterate over the document comparisons
for comparison in jaccard_list:
    print("The jaccard similarity between '{}' and '{}' is '{:.4f}'".format(
        comparison.label_a, 
        comparison.label_b, 
        comparison.jaccard_similarity
    ))

# retrieve the documents with the max jaccard similarity
max_sim_comp = max(jaccard_list, key=attrgetter('jaccard_similarity'))
print("The documents with the highest jaccard similarity '{:.4f}' are '{}' and '{}'".format(
    max_sim_comp.jaccard_similarity, 
    max_sim_comp.label_a, 
    max_sim_comp.label_b
))

print("\nQuestion 2", "-"*70)
print("a)\tGenerate the document vectors for the scientific texts using one-hot representations.")

# unpack the nested lists in file_token_list and take the set to get the full vocab
combined_vocabulary = set(chain.from_iterable(file_token_list))

# make an empty dictionary from the vocabulary
# since python dictionaries are orderless, we need to use an OrderedDict
# to keep the token order consistent between documents
vocab_one_hot_dict = OrderedDict((token, 0) for token in combined_vocabulary)

one_hot_dict_list = []
# iterate over document token lists
for token_list in file_token_list:
    # make a copy of the empty vocab dict
    doc_one_hot_dict = vocab_one_hot_dict.copy()

    # iterate over unique tokens in the token list
    for token in set(token_list):
        # if the token is present set its value to 1 in the dictionary
        doc_one_hot_dict[token] = 1

    # append the current doc dictionary to the one hot dict list
    one_hot_dict_list.append(doc_one_hot_dict)

# convert the dictionarys to values
count_array_list = [list(doc_one_hot_dict.values()) for doc_one_hot_dict in one_hot_dict_list]

# iterate over documents
for i, doc_label in enumerate(doc_label_list):
    print("The one-hot document vector for '{}' is:\n{}".format(
        doc_label, 
        count_array_list[i]
    ))

print("\nb)\tCompute the cosine similarity between scientificpub1 and scientificpub2.")

def cosine_similarity(list_a, list_b, epsilon: float = 1e-5):
    # based on output from chatgpt
    # get the dot product of the two lists
    dot_product = np.dot(list_a, list_b)
    
    # calculate the magnitude of the lists
    # add epsilon to avoid division by 0
    magnitude_a = np.linalg.norm(list_a) + epsilon
    magnitude_b = np.linalg.norm(list_b) + epsilon
    
    # calculate the cosine similarity
    cosine_sim = dot_product / (magnitude_a * magnitude_b)
    return cosine_sim

doc1_count_vector, doc2_count_vector, doc3_count_vector = count_array_list

cosine_sim_12 = cosine_similarity(doc1_count_vector, doc2_count_vector)
cosine_sim_23 = cosine_similarity(doc2_count_vector, doc3_count_vector)
print(f"The cosine similarity between 'scientificpub1' and 'scientificpub2' is '{cosine_sim_12}'")
print(f"The cosine similarity between 'scientificpub2' and 'scientificpub3' is '{cosine_sim_23}'")