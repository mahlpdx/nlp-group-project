from rouge import Rouge
from collections import namedtuple
"""
Functions to evaluate generated and reference
summaries with R.O.U.G.E metrics:

Recall Oriented Understudy for Gisting Evaluation

Rouge is essentially comparing the match-rate of n-grams between our resultant model and the original text.

Authors:
Jingjing Zhao
John Lorenz IV
"""

class Evaluation():
    """
    Utility class for computing F1, Precision, and Recall metrics for n-grams of summarization models.
    """
    def __init__(self):
        pass

    def evaluation(self, summary, original):
        rouge = Rouge()
        return rouge.get_scores(summary, original)

    def score_display(self, rouge_scores):
        iters = [1, 2, 3, 'l']
        for i in iters:
            idx = 'rouge-'+str(i)
            try:
                rouge_ngram = rouge_scores[0][idx]
                print('#'*5, str(i)+'-gram metrics:','#'*5)
                print('Recall:', rouge_ngram['r'],'\nF1 Score:', rouge_ngram['f'], '\nPrecision:', rouge_ngram['p'], '\n')      
            except KeyError:
                print(str(i)+'-gram metrics not available for this example.')

RougeStats = namedtuple("RougeStats", "r1 p1 f1 r2 p2 f2 rl pl fl")

if __name__ == '__main__':
    # To calculate scoring metrics, repeat the below using your own strings
    eval = Evaluation()
    generated_summary = ['My name is jacob and I like water']
    reference_summary = ['My title is jacob and I enjoy drinking']
    scores = eval.evaluation(generated_summary,reference_summary)
    eval.score_display(scores)
    
