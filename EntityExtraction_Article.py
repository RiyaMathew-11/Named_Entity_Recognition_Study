# Entity extraction from an article.
from collections import Counter

from bs4 import BeautifulSoup
import requests
import re

from spacy import displacy

from SpaCy import nlp

# install html5lib before execution
# hmtl5lib is a python parsing library for parsing HTML
def url_to_string(url):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

#Taking an article from New York times as our base reference

string_format = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(string_format)
print("Number of Entities in the article: ",len(article.ents))

# To find the distribution of labels
label_names = [entity.label_ for entity in article.ents ]
print(Counter(label_names))

# Counter holds the data in an unordered collection, just like hashtable objects. The elements here represent the keys and the count as values

# To find the most frequent tokens
freq_tokens = [tokens.text for tokens in article.ents]
print("\n\nThe most common tokens are ",Counter(freq_tokens).most_common(3))

sentences = [x for x in article.sents]
print(sentences)

displacy.render(nlp(str(sentences)), jupyter=True, style='ent')