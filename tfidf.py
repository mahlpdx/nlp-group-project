import nltk
"""
Functions to perform tf, idf for corpora.
Functions to extract best summary from a document.
"""

def tf(word, document):
    """Calculate the tf of a word
    Inputs:
        word (str): word to calculate tf of
        document (str): document containing word

    Return:
        (float): tf value
    """


def idf(documents, word):
    """Calculate the idf for a word

    Inputs:
        word (str): word
        documents (list<str>): all documents

    Return:
        (float): idf value
    """
    return 1


def generate_terms(document, n=1):
    """Generate all single and multi-word terms needed
    to generate tf-idf matrix

    From paper: `Generate multi-word terms of length 1 to TL as follows. 
    For a document with DL words, there are DL single-word terms, DL 
    - 1 two word terms, and so on. Finally, we have DL - TL + 1 terms
    with TL words.`

    Inputs:
        n (int): maximumal number of words in a term
        document (str): document containing word

    Return:
        list<str>: list of all single and multi-word terms
    """
    return [""]


def idf_mapping(documents):
    """Create mapping for all terms to their idf scores

    Inputs:
        documents (list<str>): all documents in corprus   

    Return:
        {string: int}: mapping term -> idf value
    """
    return {}


def rank_sequences(document):
    """Rank all sequences of up to 1000 (we can make this smaller) words. 
    Returned sequency is text summary.

    Inputs:
        document (str): document containing word

    Return:
        (str): highest ranked sequence AKA the summary

    """
    return ""


if __name__ == '__main__':
    # Run test cases here
    print(1)
