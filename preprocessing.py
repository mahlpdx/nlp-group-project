import nltk
import string
import re
import csv

"""
Functions to read in corpora.
Functions to split corpora into train, validation (?), 
and test sets.
Functions to preprocess documents.
"""

CORPUS_FILE_PATH = "D:\Desktop\CS 410 Natural Language Processing\Project\linux.csv"


def readCorpora(corpora):
    """Reads in the corpora.

    Loops through the dictionary and calls preprocess on each document, 
    storing the processed documents in a new dictionary.



    Inputs:
        corpora (str): file path

    Return:
        (dic): dictionary of all processed documents

    """

    reader = csv.DictReader(
        open(corpora, "rU", encoding='cp932', errors='ignore'))
    temp = {}
    for row in reader:
        temp.update({row["man_entry"]: row["tldr_summary"]})

    print(temp[:5])


def splitData(data):
    """Splits data into train, validation (?), and test sets.

    Uses nltk to split the data into train, and test sets.

    Inputs:
        data (dic): dictionary of all processed documents

    Return:
        (dic): split data

    """


def removeSpecials(document):
    """Removes special characters from string
    """

    return [x for x in document if x not in string.punctuation]


def removePhoneNumbers(document):
    """Removes phone numbers from string
    """

    return re.sub(r"((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))", "", document)


def removeEmails(document):
    """Removes all emails from string
    """

    return re.sub(r"[a-z0-9]+@[a-z]+\.[a-z]{2,3}", "", document)


def removeStopWords(document, stopWords):
    """Filters stop words out of document
    """

    return [x for x in document if x not in stopWords]


def preprocess(document):
    """Preprocess a single document.

    Splits text, tokenizes, removes specials, phone numbers, emails.
    Uses custom stopword list.

    Inputs:
        Document (str): word document

    Return:
        (str): processed document

    """
    x = removeEmails(document)
    x = removePhoneNumbers(x)
    y = nltk.tokenize.word_tokenize(x)
    y = removeSpecials(y)

    stop_words = [
        "and", "the", "is", "are", "this", "at", "of", "to", "in", "on",
        "for", "or", "a", "an", "as", "page", "by", "with", "our", "we", "that",
        "may"
    ]

    y = removeStopWords(y, stop_words)

    return y


if __name__ == '__main__':
    # Run test cases here
    test_string = "test@hotmail.org is dog the (555) 555-5555 this n#kj&sal9(! 555-555-5555"
    test_string_two = "dog (555) 555-5555 test@hotmail.org n#kj&sal9(! test@hotmail.org"
    """ print(removeEmails(test_string))
    print(removeEmails(test_string_two))
    print(removePhoneNumbers(test_string))
    print(preprocess(test_string)) """
    readCorpora(CORPUS_FILE_PATH)
    # print(1)
