from preprocessing import *
from evaluation import *
from tfidf import *
import tensorflow_datasets as tfds
import tensorflow as tf
from collections import namedtuple
import numpy as np

"""This is main script used to run the pipeline

Figure from paper:

Document Corpus --> Preprocessing --> multi-word terms --> compute tf-idf
                           |                                     /
                           |                                    /     
                           V                                   /
                   generate candidate sequences               /
                         \                                   /
                          \                                 /     
                           \                               /
                            > TF-IDF scores for sequences <
                                         |
                                         |
                                         V
                                Best scored summary                         
"""

RougeStats = namedtuple("RougeStats", "r1 p1 f1 r2 p2 f2 rl pl fl")

if __name__ == '__main__':
    # 1.  Load dataset
    train, test = tfds.load(
        'multi_news',
        split=['train[:500]', 'test[:100]']
    )
    train = [
        (ex['document'].numpy().decode("utf-8"),
        ex['summary'].numpy().decode("utf-8") )
        for ex in list(train)
    ]
    test = [
        (ex['document'].numpy().decode("utf-8"),
        ex['summary'].numpy().decode("utf-8") )
        for ex in list(test)
    ]

    # 2. Preprocess documents to create text corpus
    
    # Each sample contains multiple news articles so
    # we arbitrarily select the first one
    train_corpus = [
        preprocess(ex[0].split("|||||")[0]) for ex in train
    ]
    test_corpus = [
        preprocess(ex[0].split("|||||")[0]) for ex in test
    ] 
    
    # 3. Assign term size to be used for corpus
    term_size = 1

    # 4. Load mapping of term to idf
    print ("Processing idf_map...")
    idf_map = idf_mapping(train_corpus, term_size)
    print ("Completed idf map!")

    # 5. Loop through test set to generate summaries
    print ("Generating summaries...")

    # Assign test size (number of samples)
    test_size = 50
    summary_sizes = [100, 200, 300]
    for summary_size in summary_sizes:
        predicted_summaries = []
        print ("Running summary generation for summary size of {}".format(summary_size))
        for i in range(test_size):
            tf_map = tf_mapping(test_corpus[i], term_size)
            predicted_summaries.append(
                generate_summary(test_corpus[i], tf_map, idf_map, summary_size, term_size)
            )
        
        print ("Completed summaries!")
        print ("Metric evaluation...")
        
        # 6.  Apply ROUGE evaluation to generated summaries
        # Currently only applying to a single sampls
        eval = Evaluation()
        all_scores = []
        for i in range(len(predicted_summaries)):
            print (i)
            try:
                score = eval.evaluation(summary_str(predicted_summaries[i]), test[i][1])[0]
                all_scores.append(
                    RougeStats(score['rouge-1']['r'], score['rouge-1']['p'], score['rouge-1']['f'],
                    score['rouge-2']['r'], score['rouge-2']['p'], score['rouge-2']['f'],
                    score['rouge-l']['r'], score['rouge-l']['p'], score['rouge-l']['f']  
                    ) 
                )
            except:
                print ("Could not add index {} due to Summary length".format(i))

        average_scores = RougeStats(
            np.average([s.r1 for s in all_scores]),
            np.average([s.p1 for s in all_scores]),
            np.average([s.f1 for s in all_scores]),
            np.average([s.r2 for s in all_scores]),
            np.average([s.p2 for s in all_scores]),
            np.average([s.f2 for s in all_scores]),
            np.average([s.rl for s in all_scores]),
            np.average([s.pl for s in all_scores]),
            np.average([s.fl for s in all_scores]),
        )
        
        print ("Average scores for summary size of {} and test size of {}:".format(summary_size, test_size))
        print (average_scores)
        print ("Completed evaluation for summary size of {}".format(summary_size))
    
    print ("Completed all evaluation!")

