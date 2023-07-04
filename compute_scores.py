import math
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import math


def get_text_score(text, tokenizer, lm):
    input_id = tokenizer.encode(text, add_special_tokens=False)
    input_id_to_tensor = torch.tensor([input_id]).cuda()
    # input_id_to_tensor = torch.tensor([input_id])
    predictions = lm(input_id_to_tensor)
    predictions = predictions.logits
    softmax = nn.Softmax(dim=1)
    predictions = softmax(predictions)
    log_p = []
    for i in range(0, predictions.shape[1]):
        log_p.append(math.log(predictions[0][i][input_id[i]]))
    return sum(log_p) / len(log_p)


def get_score(texts, lm, tokenizer):
    log_scores = []
    for text in texts:
        log_scores.append(get_text_score(text, tokenizer, lm))
    print(log_scores)
    mean = sum(log_scores[1:]) / (len(texts) - 1)
    d = log_scores[0] - mean
    variation = 0
    for log_score in log_scores:
        variation += (log_score - mean) ** 2
    variation = variation / (len(texts) - 1)
    print(d / math.sqrt(variation))
    return d / math.sqrt(variation)


def predict(texts, model, epsilon):
    d, variation = get_score(texts, model)
    print(d / math.sqrt(variation))
    if d / math.sqrt(variation) > epsilon:
        return True
    else:
        return False
