"""
Authors: Austin Brown, Tim Hall
Date: 11/25/2022
NLP Final Project
"""

import nltk
import string
import re
import csv
import random
import tensorflow_datasets as tfds


"""
Functions to read in corpora.
Functions to split corpora into train, validation (?), 
and test sets.
Functions to preprocess documents.
"""

nltk.download('punkt')


def removeSpecials(document):
    """Removes special characters from string
    Inputs:
        document list<str>: Tokenized document
    Returns:
        list<str>: Tokenized list w/o special characters
    """

    return [x for x in document if x not in string.punctuation]


def removePhoneNumbers(document):
    """Removes phone numbers from string
    Inputs:
        document str: document in string format
    Returns:
        str: document w/o phone numbers
    """

    return re.sub(r"((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))", "", document)


def removeEmails(document):
    """Removes all emails from string
    Inputs:
        document str: document in string format
    Returns:
        str: document w/o emails
    """

    return re.sub(r"[a-z0-9]+@[a-z]+\.[a-z]{2,3}", "", document)


def removeStopWords(document, stopWords):
    """Filters stop words out of document
    Inputs:
        document list<str>: Tokenized list
        stopWords list<str>: Stop words
    Returns:
        list<str>: document w/o stop words
    """

    return [x for x in document if x.lower() not in stopWords]


def preprocess(document):
    """Preprocess a single document.
    Removes Emails and phone numbers, tokenizes and removes special characters and 
    stop words. (DO NOT change order)

    Inputs:
        Document (str): word document

    Return:
        list<str>: processed document

    """
    stop_words = [
        "and", "the", "is", "are", "this", "at", "of", "to", "in", "on",
        "for", "or", "a", "an", "as", "page", "by", "with", "our", "we", "that",
        "may", "if"
    ]

    x = removeEmails(document)
    x = removePhoneNumbers(x)
    y = nltk.tokenize.word_tokenize(x)
    y = removeSpecials(y)

    return y


def load_multi_news(train_size, test_size):
    """Load and preprocess multi news dataset

    """
    # 1.  Load dataset
    train, test = tfds.load(
        'multi_news',
        split=[
            'train[:{}]'.format(train_size),
            'test[:{}]'.format(test_size)
        ]
    )

    train_documents = [
        preprocess(
            ex['document'].numpy().decode("utf-8").split("|||||")[0]
        )
        for ex in list(train)
    ]

    train_summaries = [
        preprocess(
            ex['summary'].numpy().decode("utf-8").split("|||||")[0]
        )
        for ex in list(train)
    ]

    test_documents = [
        preprocess(
            ex['document'].numpy().decode("utf-8").split("|||||")[0]
        )
        for ex in list(test)
    ]

    test_summaries = [
        preprocess(
            ex['summary'].numpy().decode("utf-8").split("|||||")[0]
        )
        for ex in list(test)
    ]

    return train_documents, train_summaries, test_documents, test_summaries


if __name__ == '__main__':
    # Run test cases here
    test_string = "test@hotmail.org is dog the (555) 555-5555 this n#kj&sal9(! 555-555-5555"
    test_string_two = "dog (555) 555-5555 test@hotmail.org n#kj&sal9(! test@hotmail.org"
    print(removeEmails(test_string))
    print(removeEmails(test_string_two))
    print(removePhoneNumbers(test_string))
    print(preprocess(test_string))
    train_documents, train_summaries, test_documents, test_summaries = load_multi_news(
        10, 10
    )
