import numpy as np
"""
Functions to perform tf, idf for corpora.
Functions to extract best summary from a document.

Authors:
Drew Mahler
Al Khatab Rashdi
"""


def tf(term, document):
    """Calculate the tf of a term
    Inputs:
        token (str): term to calculate tf of
        document (list<str>): document as a list of tokens

    Return:
        (float): tf value
    """
    term_length = len(term.split(" "))
    document_length = len(document)
    term_appearances = len(" ".join(document).split(term)) - 1
    return term_appearances / (document_length - term_length + 1)

# IDF = > Inverse document frequency


def idf(documents, term):
    """Calculate the idf for a term

    Inputs:
        term (str): token, possibly consisting of multiple words
        documents (list<list<str>>): all documents

    Return:
        (float): idf value
    """
    num_documents = len(documents)
    term_documents = 0
    for document in documents:
        if term in document:
            term_documents += 1

    return np.log((num_documents + 1) / (term_documents + 1))


def generate_terms(document, n):
    """Generate all single and multi-word terms needed
    to generate tf-idf matrix

    From paper: `Generate multi-word terms of length 1 to TL as follows. 
    For a document with DL words, there are DL single-word terms, DL 
    - 1 two word terms, and so on. Finally, we have DL - TL + 1 terms
    with TL words.`

    Inputs:
        n (int): maximal number of words in a term
        document (list<str>): document as a list of tokens

    Return:
        list<str>: list of all single and multi-word terms
    """
    terms = set()
    if n == 1:
        return set(document)
    elif n == 2:
        for idx in range(1, len(document)):
            terms.add("{} {}".format(document[idx-1], document[idx]))
    elif n == 3:
        for idx in range(2, len(document)):
            terms.add("{} {} {}".format(
                document[idx-2], document[idx-1], document[idx])
            )
    else:
        raise ("Max term allowed is 3")

    return terms


def tf_mapping(document, max_term_size):
    """Create mapping for all terms to their df scores for a given document

    Inputs:
        documents (list<str>): single document as a list of tokens
        max_term_size (int): maximum number of tokens in a term   

    Return:
        dict[string -> float]: mapping term -> df value
    """
    tf_map = {}
    all_terms = set()
    for i in range(1, max_term_size+1):
        all_terms.update(
            generate_terms(document, n=i)
        )

    for term in all_terms:
        tf_map[term] = tf(term, document)

    return tf_map


def idf_mapping(documents, max_term_size):
    """Create mapping for all terms to their idf scores

    Inputs:
        documents (list<list<str>>): all documents in corprus
        max_term_size (int): maximum number of tokens in a term 

    Return:
        dict[string -> float]: mapping term -> idf value
    """
    idf_map = {"--test-only": np.log(len(documents)/1)}
    # The above is used to handle the case where word is in test corpus
    # but not in the training corpus

    all_terms = set()
    for document in documents:
        for i in range(1, max_term_size+1):
            all_terms.update(generate_terms(document, n=i))

    for term in all_terms:
        idf_map[term] = idf(documents, term)

    return idf_map


def generate_sequences(document, seq_size):
    """Generate candidate sequences for the summaries.

    Inputs:
        document (list<str>): individual document to summarize
        seq_size (int): maximum number of tokens in a term 

    Return:
        list<str>: candidate summaries

    """
    sequences = []
    for idx in range(seq_size, len(document)):
        sequences.append(document[idx-seq_size:idx])
    return sequences


def tfidf(term, idf_map, tf_map):
    """Calculate tfidf score for a given term

    Inputs:
        term (str): single or multi-word term
        idf_map (dict[string -> float]): mapping term -> idf value
        tf_map (dict[string -> float]): mapping term -> tf value

    Return:
        (float): total tfidf score
    """
    if term in idf_map:
        return tf_map[term] * idf_map[term]
    else:
        return tf_map[term] * idf_map["--test-only"]

def baseline_summary(document, seq_pct):
    """Abitrarily chooses the first seq_pct * document_length words
    
     Inputs:
        document (list<str>): document containing tokens
        seq_pct (float): length of continuous sequences as percent of doc length

    Return:
        (str): baseline summary

    """
    seq_size = int(np.ceil(seq_pct * len(document)))
    return document[0:seq_size]

def generate_summary(document, tf_map, idf_map, seq_pct, max_term_size):
    """Rank all sequences of up to seq_size  words. 
    Returned sequency is text summary.

    Inputs:
        document (list<str>): document containing tokens
        tf_map (dict[string -> float]): mapping term -> tf value
        idf_mapping (dict[string -> float]): mapping term -> idf value
        seq_pct (float): length of continuous sequences as percent of doc length
        max_term_size (int): maximum number of tokens in a term

    Return:
        (str): highest ranked sequence AKA the final summary

    """
    seq_size = int(np.ceil(seq_pct * len(document)))
    best_score = previous_score = 0
    best_sequence = None
    # Loop through each possible sequence
    for idx in range(seq_size, len(document)+1):
        # Identify candidate sequence
        candidate_sequence = document[idx-seq_size:idx]
        # First sequence needs to calculate all tf-idf scores
        if idx == seq_size:
            candidate_score = 0
            for term_size in range(1, max_term_size+1):
                # Keep track of position
                pos = 0
                while pos+term_size < seq_size:
                    term = " ".join(candidate_sequence[pos:pos+term_size])
                    candidate_score += tfidf(term, idf_map, tf_map)
                    pos += term_size
            best_sequence = candidate_sequence
            best_score = previous_score = candidate_score

        # All other sequences subtract terms from beginning and add terms from end
        else:
            candidate_score = previous_score
            for term_size in range(1, max_term_size+1):
                # Subtract terms containing the starting words
                old_term = " ".join(
                    document[idx-seq_size-1:idx-seq_size-1+term_size])
                candidate_score -= tfidf(old_term, idf_map, tf_map)
                # Add terms containig the ending words
                new_term = " ".join(candidate_sequence[-term_size::])
                candidate_score += tfidf(new_term, idf_map, tf_map)

            if candidate_score > best_score:
                best_score = candidate_score
                best_sequence = candidate_sequence
            previous_score = candidate_score

    return best_sequence


def summary_str(s):
    """
    Inputs:
        s (list[str]): token representation of summary

    Returns:
        (str): string representation of summary
    """
    summary = ' '.join(s)
    return summary


if __name__ == '__main__':
    # Run test cases here
    documents = [
        ["I", "am", "a", "star", "I", "am", "everything",
            "that", "I", "want", "to", "be"],
        ["The", "test", "is", "hard", "but", "I",
            "am", "what", "I", "doggy", "kitty", "kookoo"],
        ["What", "is", "is", "hard", "but", "is",
            "is", "what", "that", "dog", "to", "be"],
        ["Crazy", "is", "blood", "not", "water", "she",
            "is", "what", "that", "dog", "to", "be"],
    ]
    # TF Example
    print(tf("I", documents[0]))

    # Generate terms of length 1
    print(generate_terms(documents[0], n=1))

    # Generate terms of length 2
    print(generate_terms(documents[0], n=2))

    # Generate terms of length 3
    print(generate_terms(documents[0], n=3))

    # Generate candidate summary sequences of a certain size
    print(generate_sequences(documents[0], 4))

    # Generate idf mapping
    idf_map = idf_mapping(documents, 3)

    # Generate tf mapping for a given document
    tf_map = tf_mapping(documents[1], 3)

    # Identify top ranking sequence as summary
    summary = generate_summary(documents[1], tf_map, idf_map, 0.5, 3)
    print(summary)

    # join summary strings into one string to create a sentence
    s = summary_str(summary)

    print(s)
