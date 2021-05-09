from pprint import pprint

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

# Using the same sentence as before.
# In spacy, need only apply nlp once, the entire background pipeline will return the objects afterwards.

doc = nlp('Google has used Android as a vehicle to cement the dominance of its search engine,” said Margrethe Vestager, Europe’s antitrust chief.')
print("\nEntity level recognition:\n")
pprint([(X.text, X.label_) for X in doc.ents])

# Irregularity - Android is taken as an organisation, while Google has been ignored

print("\nBILUO Taaging scheme to describe entity boundaries")
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])

