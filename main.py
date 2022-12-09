from preprocessing import *
from evaluation import *
from tfidf import *
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
"""TO CHANGE EXPERIMENTAL SETUP MODIFY THE FOLLOWING

TRAIN_SIZE: Size of the training set
TEST_SIZE: Size of the test set
TERM_SIZE: Size of the multi-word term
SUMMARY_PCT: Percentage of total document length to use for summary
BASELINE_ON: If set to true, calculate a baseline summary. If
            set to false, calculate summary via tf-idf method. 
"""
TRAIN_SIZE = 500
TEST_SIZE = 2000
TERM_SIZE = 2
SUMMARY_PCT = 0.1
BASELINE_ON = False

if __name__ == '__main__':
    # Experimental setup
    print("""
    EXPERIMENT SETUP:
        Training Size: {}
        Test Size: {}
        Term Size: {}
        Summary Percent: {}
    """.format(TRAIN_SIZE, TEST_SIZE, TERM_SIZE, SUMMARY_PCT))

    # 1. Load dataset
    train_documents, train_summaries, test_documents, test_summaries = load_multi_news(
        TRAIN_SIZE, TEST_SIZE
    )

    # 2. Load mapping of term to idf
    print("Processing idf_map...")
    idf_map = idf_mapping(train_documents, TERM_SIZE)
    print("Completed idf map!")

    # 3. Generate summaries of given size
    print("Generating summaries...")
    predicted_summaries = []
    for idx in range(TEST_SIZE):
        tf_map = tf_mapping(test_documents[idx], TERM_SIZE)
        if BASELINE_ON:
            predicted_summaries.append(
                baseline_summary(test_documents[idx], SUMMARY_PCT)
            )
        else:
            predicted_summaries.append(
                generate_summary(
                    test_documents[idx],
                    tf_map,
                    idf_map,
                    SUMMARY_PCT,
                    TERM_SIZE
            )
        )
    print("Completed summaries...")

    # 4. Apply ROUGE evaluation to generated summaries
    print("Metric evaluation...")
    eval = Evaluation()
    all_scores = []
    for idx in range(len(predicted_summaries)):
        print(idx)
        score = eval.evaluation(
            summary_str(predicted_summaries[idx]),
            summary_str(test_summaries[idx])
        )[0]
        all_scores.append(
            RougeStats(
                score['rouge-1']['r'],
                score['rouge-1']['p'],
                score['rouge-1']['f'],
                score['rouge-2']['r'],
                score['rouge-2']['p'],
                score['rouge-2']['f'],
                score['rouge-l']['r'],
                score['rouge-l']['p'],
                score['rouge-l']['f']
            )
        )

    # Determine averages for all metrics
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

    print("Average scores for summary percentage of {} and test size of {}:".format(
        SUMMARY_PCT, TEST_SIZE
    ))
    print(average_scores)
    print("Experiment complete!")
