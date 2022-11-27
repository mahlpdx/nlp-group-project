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


"""
Functions to read in corpora.
Functions to split corpora into train, validation (?), 
and test sets.
Functions to preprocess documents.
"""

CORPUS_FILE_PATH = "linux.csv"


def readCorpora(corpora):
    """Reads in the corpora.

    Loops through the dictionary and calls preprocess on each document, 
    storing the processed documents in a new dictionary.

    man_entry,tldr_summary  

    Inputs:
        corpora (str): file path

    Return:
        (list<str>, str): processed corpora (tokens, summary)

    """
    output = []
    csv.field_size_limit(csv.field_size_limit() * 10)
 
    with open(CORPUS_FILE_PATH, encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        temp = []
        count = 0
        for row in reader:
            if not row["man_entry"]:
                count += 1
                continue
            temp = preprocess(row["man_entry"])
            output.append((temp, row['tldr_summary']))
    
        print("number of empty man_entry:", count)
        return output


def splitData(data, train=.75, mixup=True):
    """Splits data into train, validation (?), and test sets.

    Inputs:
        data list<(list<str>, str)>: processed documents
        train float: [0, 1] percentage of dataset to use as train set
        mixup bool: Shuffle the dataset

    Return:
        (list<(tokens, summary)>), (list<(tokens, summary)>) train, test

    """
    temp = data

    if mixup:
        random.shuffle(temp)
    
    x = int(len(temp) * train)
    return temp[:x], temp[x:]

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
    data = readCorpora(CORPUS_FILE_PATH)
    print("len of data", len(data))
    train, test = splitData(data)

    print("len of train:", len(train))
    print("len of test:", len(test))