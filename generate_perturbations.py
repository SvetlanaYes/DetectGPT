import random
import torch


def decode_text(input_id, prediction, tokenizer, mask_token):
  output_id = []
  for i, id_ in enumerate(input_id):
        if id_ == mask_token:
           output = torch.argmax(prediction.logits[0][i])
           output_id.append(output)
        else:
           output_id.append(id_)
  decoded_output = tokenizer.decode(output_id)
  return decoded_output


def generate_with_bert(text, tokenizer, lm, mask_token):
    input_id = tokenizer.encode(text, add_special_tokens=False)
    input_id_to_tensor = torch.tensor([input_id]).cuda()
    prediction = lm(input_id_to_tensor)
    decoded_output = decode_text(input_id, prediction, tokenizer, mask_token)
    return decoded_output


mask_word = {
    "bert-base-multilingual-cased": ["[MASK]", 103],
    "xlm-roberta-large": ["<mask>", 250001],
    "cointegrated/rut5-base-multitask": " ___ ",
    "blinoff/roberta-base-russian-v0": ["<mask>", 4]
}


mask_model = {
    'DeepPavlov/rubert-base-cased':generate_with_bert,
    'bert-base-multilingual-cased': generate_with_bert,
    'xlm-roberta-large': generate_with_bert,
    # 'cointegrated/rut5-base-multitask': generate_with_t5,
    'blinoff/roberta-base-russian-v0': generate_with_bert,
}


# def generate(text, model, **kwargs):
#     tokenizer = T5Tokenizer.from_pretrained(model)
#     model = T5ForConditionalGeneration.from_pretrained(model)
#     inputs = tokenizer(text, return_tensors='pt')
#     with torch.no_grad():
#         hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
#     return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


def generate_indexes(size):
    indexes = []
    count_of_words_to_change = size * 15 // (100 * 2)
    while len(indexes) != count_of_words_to_change:
        n = random.randint(0, size - 2)
        indxs = [n - 2, n - 1, n, n + 1, n + 2]
        if len(set(indxs).difference(indexes)) == len(indxs):
            indexes.append(n)
    return indexes


def get_perturbations(text, model, tokenizer, lm):
    splt_text_ = text.split(" ")
    size = len(text.split(" "))
    indexes = generate_indexes(size)
    pattern = mask_word[model][0]
    for i in indexes:
      splt_text_[i] = pattern
      splt_text_[i + 1] = pattern
    masked = ' '.join(splt_text_)
    gen_text = mask_model[model](masked, tokenizer, lm, mask_word[model][1])
    return gen_text
