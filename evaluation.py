from rouge import Rouge
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
"""
Functions to evaluate generated and reference
summaries with R.O.U.G.E metrics:

Recall Oriented Understudy for Gisting Evaluation

Rouge is essentially comparing the match-rate of n-grams between our resultant model and the original text.

------------
BLEU metric
https://aclanthology.org/P02-1040.pdf

Measures the 'quality' of the text translation.

Authors:
Jingjing Zhao
John Lorenz IV
"""
SAMPLE_SUMMARY = ["Sally sells seashells.", "Sally is by the seashore.", "Sally likes the seashore.", "Seashells are collected by Sally."]
SAMPLE_REFERENCES = [["Sally sells seashells by the sea shore.", "Sally has many seashells that she sells."], ["How many shells did Sally sell by the Seashore", "Do you think Sally will sell me some seashells?"]]

def make_contiguous(list_of_list):
    big_str = ''
    for x in list_of_list:
        big_str += ' '.join(x)
        big_str += ' '
    return big_str

class Evaluation():
    """
    Utility class for computing F1, Precision, and Recall metrics for n-grams of summarization models.
    """
    def __init__(self):
        pass

    def apply_sacrebleu(self, summary, original):
        sacrebleu = BLEU()
        score = sacrebleu.corpus_score(summary, original)
        print('SACREBLEU:', score)

    def apply_bleu(self, summary, original, ngrams=None, manual_weight=None):
        #print(summary)
        #print(original)
        smoothing = SmoothingFunction().method1
        if ngrams and not manual_weight:
            w = self._apply_bleu_ngram(ngrams)
            score = sentence_bleu(original, summary, w, smoothing_function=smoothing)
            print(str(ngrams)+'-gram BLEU score:', score)
        elif manual_weight:
            score = sentence_bleu(original, summary, manual_weight, smoothing_function=smoothing)
            print('Cumulative', str(ngrams)+' score:', score)
        else:
            score = sentence_bleu(original, summary, smoothing_function=smoothing)
            print('BLEU score:', score)

    def _apply_bleu_ngram(self, ngrams):
        weight_tuple = [0, 0, 0, 0]
        weight_tuple[ngrams-1] = 1 # one-hot encoding
        weight_tuple = set(weight_tuple)
        return weight_tuple
        
    def apply_rouge(self, summary, original):
        print('\n----> SUMMARY:', summary)
        print('----> REFERENCE TEXT:', original)
        rouge = Rouge()
        return rouge.get_scores(summary, original)

    def score_display(self, rouge_scores):
        iters = [1, 2, 'l']
        for i in iters:
            idx = 'rouge-'+str(i)
            try:
                rouge_ngram = rouge_scores[0][idx]
                print('#'*5, str(i)+'-gram metrics:','#'*5)
                print('Recall:', rouge_ngram['r'],'\nF1 Score:', rouge_ngram['f'], '\nPrecision:', rouge_ngram['p'], '\n')      
            except KeyError:
                print(str(i)+'-gram metrics not available for this example.')

    

if __name__ == '__main__':
    # To calculate scoring metrics, repeat the below using your own strings
    eval = Evaluation()
    generated_summary = ['My name is jacob and I like water']
    reference_summary = ['My title is jacob and I enjoy drinking']
    scores = eval.apply_rouge(SAMPLE_SUMMARY[0], ' '.join(SAMPLE_REFERENCES[0]))
    eval.score_display(scores)
    #make_contiguous(SAMPLE_REFERENCES)
    contiguous_summary = SAMPLE_SUMMARY[0]
    contiguous_reference = make_contiguous(SAMPLE_REFERENCES)

    eval.apply_bleu(contiguous_summary, contiguous_reference)
    eval.apply_bleu(contiguous_summary, contiguous_reference, 2)
    # weights MUST add to 1 and be evenly spread
    # E.G cumulative trigram weight = (0.33, 0.33, 0.33, 0)
    # bigram = (0.5, 0.5, 0, 0)
    eval.apply_bleu(contiguous_summary, contiguous_reference, ngrams = 2, manual_weight=(0.50, 0.50, 0, 0))
    eval.apply_sacrebleu(SAMPLE_SUMMARY, SAMPLE_REFERENCES) # List summary, and list of list reference
    eval.apply_sacrebleu(SAMPLE_SUMMARY[0], ' '.join(SAMPLE_REFERENCES[0])) # Two strings

