# Named Entity Recognition can be considered as the first step towards extracting information that classifies named
# entities in text into well-defined categories -> name of places, persons, organisation, locations etc., 

# Building a Named Entity Recogniser using NLTK

# Import nltk functionalities for tokenization and POS Tagging

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

sample = "Google has used Android as a vehicle to cement the dominance of its search engine,” said Margrethe Vestager, Europe’s antitrust chief."


# Function to tokenize and assign tags
def wrangle(sentence):
    tokenizer = nltk.RegexpTokenizer(r"\w+")    # Removing punctuation
    words = tokenizer.tokenize(sentence)
    new_words = nltk.pos_tag(words)
    return new_words


processed_sample = wrangle(sample)
for element in processed_sample:
    print(element)

# Implement noun phrase chunking to identify named entities using a regular expression
# Chunking - process of taking individual pieces of information and grouping them into larger units.
print("\n Chunks: ")
pattern = 'NP: {<DT>?<JJ>*<NN>}'

chunker = nltk.RegexpParser(pattern)
chunks = chunker.parse(processed_sample)

for en in chunks:
    print(en)

# Create a tree representation
#chunks.draw()

# pprint module provides a capability to “pretty-print” arbitrary Python data structures in a form which can be used as input to the interpreter.

print("\nIOB Tags: \n")
iob_tags = tree2conlltags(chunks)
pprint(iob_tags)

print("\nNER: \n")
#ne_tree = ne_chunk(pos_tag(word_tokenize(sample)))
for word, pos, ner in iob_tags:
    print(word, pos, ner)

