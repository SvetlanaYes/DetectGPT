import csv
import sys
import argparse
import generate_perturbations
import compute_scores
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help='Path to dataset file')
    parser.add_argument("-mf", "--maskfill_model", type=str, required=True, help='mask fill model - DeepPavlov/rubert-base-cased, '
                                                                                'bert-base-multilingual-cased'
                                                                                'blinoff/roberta-base-russian-v0')

    parser.add_argument("-lm", "--language_model", type=str, required=True, help='language model - xlm-roberta-large,'
                                                                                'bert-base-multilingual-cased,'
                                                                                ' DeepPavlov/rubert-base-cased,'
                                                                                'GPT2 - small/medium/large,'
                                                                                'GPT3 - small/medium/large')
    return parser.parse_args()


def write_to_csv(csv_filename, sample):
    keys = ['id', 'doc', 'score']
    with open(csv_filename, "w", encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(keys)
        writer.writerows(zip(*[sample[key] for key in keys]))


def main(args):
    if len(sys.argv) != 7:
        return
    sample = {
      "id": [],
      "doc": [],
      "score": []
    }

    tokenizer = AutoTokenizer.from_pretrained(args.maskfill_model)
    mf = AutoModelForMaskedLM.from_pretrained(args.maskfill_model).cuda()

    generated_texts = {}
    if "gpt" in args.language_model.lower():
        print("gpt")
        tokenizer_lm = GPT2Tokenizer.from_pretrained(args.language_model)
        lm = GPT2LMHeadModel.from_pretrained(args.language_model).cuda()
    else:
        lm = AutoModelForMaskedLM.from_pretrained(args.language_model).cuda()
        tokenizer_lm = AutoTokenizer.from_pretrained(args.language_model)
    with open(args.dataset, encoding='utf-8') as f:
        csvreader = csv.reader(f, delimiter='\t')
        next(csvreader)
        for id_, doc, text in csvreader:
            text = text.strip()
            text_and_perturbations = [text]
            for i in range(100):
                  text_with_changes = generate_perturbations.get_perturbations(text, args.maskfill_model, tokenizer, mf)
                  text_and_perturbations.append(text_with_changes)
            print(id_)
            generated_texts[id_] = text_and_perturbations
            score = compute_scores.get_score(text_and_perturbations, lm, tokenizer_lm)
            sample["id"].append(id_)
            sample["doc"].append(doc)
            sample["score"].append(score)
            print(score)

    indx = args.dataset.rfind("/")
    name = args.dataset[indx + 1:indx + 6]
    with open(args.maskfill_model+ "_" + name + "_.json", "w") as f:
        json.dump(generated_texts, f)
    write_to_csv("id_doc_label_lm_" + args.language_model.replace("/", "_") + "_mf_"+args.maskfill_model.replace("/", "_") + "_" + name + ".tsv", sample)


if __name__ == "__main__":
    args = argument_parser()
    main(args)
