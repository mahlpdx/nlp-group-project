import nltk

TL = 1000 # value given in paper

def tf(word, document):
    """Calculate the tf of a word
    Inputs:
        word (str): word to calculate tf of
        document (str): document containing word

    Return:
        (float): tf value
    """

def idf(documents):
    """Calculate the idf for a word

    Inputs:
        documents (list<str>)

    Return:
        (float): idf value
    """

def generate_terms(document):
    """Generate all single and multi-word terms needed
    to generate tf-idf matrix

    From paper: `Generate multi-word terms of length 1 to TL as follows. 
    For a document with DL words, there are DL single-word terms, DL 
    - 1 two word terms, and so on. Finally, we have DL - TL + 1 terms
    with TL words.`

    Inputs:
        document (str): document containing word

    Return:
        list<str>: list of all single and multi-word terms

    """

if __name__ == '__main__':
    # Run test cases here
    print (1)