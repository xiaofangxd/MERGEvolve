from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer

def calcuate_bleu_score(tokenizer: AutoTokenizer, predicted: str, reference: str) -> float:
    smooth = SmoothingFunction()
    predicted = tokenizer.tokenize(predicted)
    reference = tokenizer.tokenize(reference)
    bleu_score = sentence_bleu(references = [reference], hypothesis = predicted, smoothing_function=smooth.method1)
    
    return bleu_score