import copy
import json
import random
import sys

sys.path.append('/data/dmekala/lost-in-the-middle/training')

import numpy as np
from datasets import load_dataset
from train import PROMPT_DICT
from nltk import sent_tokenize
from transformers import AutoTokenizer


def modify(question, answer, wiki_sample, tokenizer):
    prompt_no_input = PROMPT_DICT["prompt_no_input"]
    wiki_sentences = sent_tokenize(wiki_sample)
    sent_index = len(wiki_sentences)
    for i in range(1, len(wiki_sentences)):
        temp_wiki = " ".join(wiki_sentences[:i])
        temp_dic = {"instruction": temp_wiki + " " + question}
        sent = prompt_no_input.format_map(temp_dic) + " " + answer
        if len(tokenizer(sent)["input_ids"]) >= tokenizer.model_max_length - 100:
            sent_index = i
            break
    pruned_wiki_sample = " ".join(wiki_sentences[:sent_index])
    wiki_sentences = sent_tokenize(pruned_wiki_sample)
    question_sentences = sent_tokenize(question)
    num_wiki_sentences = len(wiki_sentences)
    num_question_sentences = len(question_sentences)
    inds = sorted(np.random.choice(list(range(1, num_wiki_sentences + 1)), num_question_sentences - 1))
    final_sentences = copy.deepcopy(wiki_sentences)
    for i, ind in enumerate(inds):
        final_sentences[ind - 1] = " ".join([final_sentences[ind - 1], question_sentences[i]])
    sample = " ".join(final_sentences) + " " + question_sentences[-1]
    return sample


def get_wiki_sample(wiki_dataset, prev_titles, question):
    for example in wiki_dataset:
        if example["title"] in prev_titles or len(sent_tokenize(example["text"])) <= len(sent_tokenize(question)):
            continue
        yield example["text"], example["title"]


if __name__ == "__main__":
    model_path = "gpt2"
    out_path = "data/gsm8k_1000_distractors.json"

    df = load_dataset("gsm8k", 'main', split="train").to_pandas()
    df_sampled = df.sample(1000).reset_index(drop=True)
    questions = list(df_sampled["question"])
    answers = list(df_sampled["answer"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = 4096

    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    out_dic_list = []
    i = 0
    prev_titles = []
    for q, a in zip(questions, answers):
        print("Running", i)
        wiki_sample, wiki_title = next(get_wiki_sample(wiki_dataset, prev_titles, q))
        prev_titles.append(wiki_title)
        new_q = modify(q, a, wiki_sample, tokenizer)
        print("Title:", wiki_title)
        print(new_q)
        print("*" * 80)
        out_dic_list.append({"instruction": new_q, "input": "", "output": a})
        i += 1

    with open(out_path, "w") as f:
        for k in out_dic_list:
            f.write(json.dumps(k, indent=4))
