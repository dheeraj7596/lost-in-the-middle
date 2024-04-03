import copy
import re
from abc import ABC, abstractmethod

import pandas as pd
from torch.utils.data import Subset
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from transformers import GPT2TokenizerFast


class PromptDataset(ABC):
    # I recommend to format the prompt completely in the dataset_cls, and provide different ways of generating prompted text:
    # # 1. original text, prompt format
    # # 2. prompt only for zero shot models
    # # 3. text, label, prompt format, see rotten tomatoes
    label_start_tag = "Labels: "
    label_end_tag = "\n"
    text_start_tag = ""
    text_end_tag = "\n"
    answer_start_tag = "Answer: "
    label_choice_names = "ABCDEFGHIJKL"

    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.prompts = DatasetTemplates(dataset_name, dataset_config_name)
        self.dataset = load_dataset(dataset_name, dataset_config_name, split=split_name)
        self.spl_token = "{{text}}"

    @abstractmethod
    def process_example(self, prompt, example):
        pass

    @abstractmethod
    def process_all_labels(self, prompt, example):
        pass


class AmazonPolarity(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name)
        self.escape_ids = []


class AppReviews(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = ["8086b434-a75e-45a4-87fb-4364601e2e05"]
        if split_name == "train":
            self.dataset = Subset(self.dataset, range(100000))

    def process_example(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].split("{{answer_choices[0]}}")[0].replace("\"{{review}}\"",
                                                                                            self.spl_token).strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": example["review"],
            "label": example["star"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["star"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["star"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class RottenTomatoes(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = []

    def process_example(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    # format prompt for zero-shot
    def process_example_with_only_prompt(self, prompt, example):
        raise NotImplementedError

    # processes an example and pad the labels as text in front of the prompts.
    # does not care about the length after tokenization.
    # expectation is with this format, training on rotten tomatoes can get 0 MSE.

    # text_proc might need some changes, but I am leaving it unchanged for now.
    #
    def process_example_with_labels_zihan(self, prompt, example, random_func_for_label_order=None):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        all_labels = list(map(lambda x: x.strip(), prompt.answer_choices.split("|||")))
        answer_index = example['label']
        if random_func_for_label_order is not None:
            answer_text = all_labels[example['label']]
            random_func_for_label_order.shuffle(all_labels)
            answer_index = all_labels.index(answer_text)
        label_str = " ".join([f"{self.label_choice_names[i]}. {all_labels[i]}" for i in range(len(all_labels))])
        label_str = f"{self.label_start_tag}{label_str}{self.label_end_tag}"
        text_str = f"{self.text_start_tag}{example['text']}{self.text_end_tag}{self.answer_start_tag}"
        answer_str = self.label_choice_names[answer_index]

        return {
            "prompt": prompt_str,
            "label_str": label_str,
            "text_str": text_str,
            "answer_str": answer_str,
        }

    def process_example_with_labels(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["label"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["label"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class Imdb(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = []

    def process_example(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_example_with_labels(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["label"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["label"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class SST(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = ["5119a0b5-5d82-4401-900a-7fafc1d48ff6", "647585d3-dac6-40c3-b6d0-f02d835ae4c4"]

    def process_example(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].replace("{{sentence}}", "{{text}}").strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": example["sentence"],
            "label": int(example["label"] > 0.5),
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_example_with_labels(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].replace("{{sentence}}", "{{text}}").strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)
        return {
            "prompt": prompt_str,
            "text": example["sentence"],
            "label": int(example["label"] > 0.5),
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = int(example["label"] > 0.5)
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["label"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class Yelp(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.prompts = DatasetTemplates(dataset_name, dataset_config_name)
        self.train_path = "/data/dheeraj/PromptConsistency/yelp/train.csv"
        self.test_path = "/data/dheeraj/PromptConsistency/yelp/test.csv"
        if split_name == "train":
            self.dataset = load_dataset("csv", data_files=self.train_path)["train"]
        elif split_name == "test":
            self.dataset = load_dataset("csv", data_files=self.test_path)["train"]
        self.spl_token = "{{text}}"
        self.escape_ids = []

    def process_example(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_example_with_labels(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["label"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["label"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class Quail(PromptDataset):
    def __init__(self, dataset_name, tokenizer, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = ["a071e73e-5fda-45b5-8a6a-b56e477a6aee", "7b0ce9fa-6aa0-4210-ab6c-1edd4b2f43df"]
        self.tokenizer = tokenizer

    def process_example(self, prompt, example):
        num_safe_tokens = 10
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        temp = prompt.jinja.split("|||")[0].strip().split("\n")
        pattern = re.compile("\{.*?\}|-|Question:|Options:|Context:|===")
        prompt_str = None
        for t in temp:
            t = t.strip()
            if re.match(pattern, t):
                continue
            prompt_str = t
            break
        if prompt_str is None:
            prompt_str = self.spl_token

        text = text_proc.replace(prompt_str, "")
        num_text_tokens = len(self.tokenizer.encode(text))
        if num_text_tokens > (self.tokenizer.model_max_length - num_safe_tokens):
            print("Chopping down tokens")
            text_copy = copy.deepcopy(text)
            question_options = text_copy.replace(example["context"], "")
            num_var_tokens = len(
                self.tokenizer.encode(question_options, max_length=self.tokenizer.model_max_length, truncation=True))
            allowed_num_tokens = self.tokenizer.model_max_length - num_var_tokens - num_safe_tokens
            if allowed_num_tokens <= 0:
                print("Allowed number of tokens less than zero")
                return None
            new_context = self.tokenizer.decode(
                self.tokenizer.encode(example["context"], max_length=allowed_num_tokens, truncation=True))
            ex_copy = copy.deepcopy(example)
            ex_copy["context"] = new_context
            text_proc, label_proc = prompt.apply(ex_copy)
            text = text_proc.replace(prompt_str, "")

        return {
            "prompt": prompt_str,
            "text": text,
            "label": example["correct_answer_id"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_example_with_labels(self, prompt, example):
        num_safe_tokens = 10
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        temp = prompt.jinja.split("|||")[0].strip().split("\n")
        pattern = re.compile("\{.*?\}|-|Question:|Options:|Context:|===")
        prompt_str = None
        for t in temp:
            t = t.strip()
            if re.match(pattern, t):
                continue
            prompt_str = t
            break
        if prompt_str is None:
            prompt_str = self.spl_token

        text = text_proc.replace(prompt_str, "")
        num_text_tokens = len(self.tokenizer.encode(text))
        if num_text_tokens > (self.tokenizer.model_max_length - num_safe_tokens):
            print("Chopping down tokens")
            text_copy = copy.deepcopy(text)
            question_options = text_copy.replace(example["context"], "")
            num_var_tokens = len(
                self.tokenizer.encode(question_options, max_length=self.tokenizer.model_max_length, truncation=True))
            allowed_num_tokens = self.tokenizer.model_max_length - num_var_tokens - num_safe_tokens
            if allowed_num_tokens <= 0:
                print("Allowed number of tokens less than zero")
                return None
            new_context = self.tokenizer.decode(
                self.tokenizer.encode(example["context"], max_length=allowed_num_tokens, truncation=True))
            ex_copy = copy.deepcopy(example)
            ex_copy["context"] = new_context
            text_proc, label_proc = prompt.apply(ex_copy)
            text = text_proc.replace(prompt_str, "")

        answer_choices_str = " [SEP] ".join(example["answers"])
        return {
            "prompt": prompt_str,
            "text": text,
            "label": example["correct_answer_id"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["correct_answer_id"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["correct_answer_id"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class AGNews(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = []

    def process_example(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].replace('{{"', "").replace('"}}', "").strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_example_with_labels(self, prompt, example):
        prompt_str = prompt.jinja.split("|||")[0].replace('{{"', "").replace('"}}', "").strip()
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)
        return {
            "prompt": prompt_str,
            "text": example["text"],
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["label"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["label"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


class DBPedia(PromptDataset):
    def __init__(self, dataset_name, dataset_config_name=None, split_name="train"):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.prompts = DatasetTemplates(dataset_name, dataset_config_name)
        self.train_path = "/data/dheeraj/PromptConsistency/dbpedia/train.csv"
        self.test_path = "/data/dheeraj/PromptConsistency/dbpedia/test.csv"
        if split_name == "train":
            self.dataset = load_dataset("csv", data_files=self.train_path)["train"]
        elif split_name == "test":
            self.dataset = load_dataset("csv", data_files=self.test_path)["train"]
        self.spl_token = "{{text}}"
        self.escape_ids = []

    def process_example(self, prompt, example):
        temp = prompt.jinja.split("|||")[0].split("{{answer_choices[0]}}")[0] \
            .replace("\"{{title}}\"", "{{text}}") \
            .replace("\"{{content}}\"", "{{text}}") \
            .replace("{{title}}", "{{text}}") \
            .replace("{{content}}", "{{text}}").strip()
        count = 0
        start = None
        end = None
        for m in re.finditer('{{text}}', temp):
            start = m.start()
            end = m.end()
            count += 1
        if count > 1:
            prompt_str = " ".join([temp[:start].strip(), temp[end:].strip()]).replace('{{"', "").replace('"}}',
                                                                                                         "").strip()
        else:
            prompt_str = temp.replace('{{"', "").replace('"}}', "").strip()

        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        return {
            "prompt": prompt_str,
            "text": " ".join([example["title"], example["content"]]),
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt = example["label"]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["label"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list

    def process_example_with_labels(self, prompt, example):
        temp = prompt.jinja.split("|||")[0].split("{{answer_choices[0]}}")[0] \
            .replace("\"{{title}}\"", "{{text}}") \
            .replace("\"{{content}}\"", "{{text}}") \
            .replace("{{title}}", "{{text}}") \
            .replace("{{content}}", "{{text}}").strip()
        count = 0
        start = None
        end = None
        for m in re.finditer('{{text}}', temp):
            start = m.start()
            end = m.end()
            count += 1
        if count > 1:
            prompt_str = " ".join([temp[:start].strip(), temp[end:].strip()]).replace('{{"', "").replace('"}}',
                                                                                                         "").strip()
        else:
            prompt_str = temp.replace('{{"', "").replace('"}}', "").strip()

        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)
        return {
            "prompt": prompt_str,
            "text": " ".join([example["title"], example["content"]]),
            "label": example["label"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }


class Race(PromptDataset):
    def __init__(self, dataset_name, tokenizer, dataset_config_name=None, split_name="train"):
        super().__init__(dataset_name, dataset_config_name, split_name)
        self.escape_ids = ["2e7f5fff-518e-4100-90f9-cca094b11e95", "1a68b62e-404c-4037-baec-7e20cb4c3f6b",
                           "af4869c4-35af-4644-86d9-27843ca4efd5", "9aedaa07-b815-4a35-890b-6100f00706aa"]
        self.tokenizer = tokenizer

    def process_example(self, prompt, example):
        num_safe_tokens = 10
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        temp = prompt.jinja.split("|||")[0].strip().split("\n")
        pattern = re.compile("\{.*?\}|-|Question:|Options:|Article:|===")
        prompt_str = None
        for t in temp:
            t = t.strip()
            if re.match(pattern, t):
                continue
            prompt_str = t
            break
        if prompt_str is None:
            prompt_str = self.spl_token

        text = text_proc.replace(prompt_str, "")
        num_text_tokens = len(self.tokenizer.encode(text))
        if num_text_tokens > (self.tokenizer.model_max_length - num_safe_tokens):
            print("Chopping down tokens")
            text_copy = copy.deepcopy(text)
            question_options = text_copy.replace(example["article"], "")
            num_var_tokens = len(
                self.tokenizer.encode(question_options, max_length=self.tokenizer.model_max_length, truncation=True))
            allowed_num_tokens = self.tokenizer.model_max_length - num_var_tokens - num_safe_tokens
            if allowed_num_tokens <= 0:
                print("Allowed number of tokens less than zero")
                return None
            new_context = self.tokenizer.decode(
                self.tokenizer.encode(example["article"], max_length=allowed_num_tokens, truncation=True))
            ex_copy = copy.deepcopy(example)
            ex_copy["article"] = new_context
            text_proc, label_proc = prompt.apply(ex_copy)
            text = text_proc.replace(prompt_str, "")

        return {
            "prompt": prompt_str,
            "text": text,
            "label": example["answer"],
            "text_proc": text_proc,
            "label_proc": label_proc
        }

    def process_example_with_labels(self, prompt, example):
        num_safe_tokens = 10
        try:
            text_proc, label_proc = prompt.apply(example)
        except Exception as e:
            print("Exception while applying prompt", e)
            return None
        temp = prompt.jinja.split("|||")[0].strip().split("\n")
        pattern = re.compile("\{.*?\}|-|Question:|Options:|Article:|===")
        prompt_str = None
        for t in temp:
            t = t.strip()
            if re.match(pattern, t):
                continue
            prompt_str = t
            break
        if prompt_str is None:
            prompt_str = self.spl_token

        text = text_proc.replace(prompt_str, "")
        num_text_tokens = len(self.tokenizer.encode(text))
        if num_text_tokens > (self.tokenizer.model_max_length - num_safe_tokens):
            print("Chopping down tokens")
            text_copy = copy.deepcopy(text)
            question_options = text_copy.replace(example["article"], "")
            num_var_tokens = len(
                self.tokenizer.encode(question_options, max_length=self.tokenizer.model_max_length, truncation=True))
            allowed_num_tokens = self.tokenizer.model_max_length - num_var_tokens - num_safe_tokens
            if allowed_num_tokens <= 0:
                print("Allowed number of tokens less than zero")
                return None
            new_context = self.tokenizer.decode(
                self.tokenizer.encode(example["article"], max_length=allowed_num_tokens, truncation=True))
            ex_copy = copy.deepcopy(example)
            ex_copy["article"] = new_context
            text_proc, label_proc = prompt.apply(ex_copy)
            text = text_proc.replace(prompt_str, "")
        choices = [a.strip() for a in prompt.answer_choices.split("|||")]
        answer_choices_str = " [SEP] ".join(choices)

        return {
            "prompt": prompt_str,
            "text": text,
            "label": example["answer"],
            "text_proc": text_proc,
            "label_proc": label_proc,
            "answer_choices_str": answer_choices_str
        }

    def process_all_labels(self, prompt, example):
        ret_list = []
        num_answers = len(prompt.answer_choices.split("|||"))
        gt_id = {"A": 0, "B": 1, "C": 2, "D": 3}
        gt = gt_id[example["answer"]]
        for a in range(num_answers):
            ex_copy = copy.deepcopy(example)
            ex_copy["answer"] = a
            dic = self.process_example(prompt, ex_copy)
            dic["gt"] = gt
            if dic is None:
                return None
            ret_list.append(dic)
        return ret_list


# NAME_TO_DATASETS = {
#     "rotten_tomatoes": RottenTomatoes("rotten_tomatoes"),
#     "quail": Quail("quail", tokenizer=tok, split_name='train[:7000]'),
#     "ag_news": AGNews("ag_news", split_name='train[:6000]'),
#
# }

# NAME_TO_TEST_DATASETS = {
#     "rotten_tomatoes": RottenTomatoes("rotten_tomatoes", split_name="test"),
    # "imdb": Imdb("imdb", split_name="test"),
    # "yelp": Yelp("yelp_review_full", split_name="test"),
    # "ag_news": AGNews("ag_news", split_name="test"),
    # "dbpedia": DBPedia("dbpedia_14", split_name="test"),
    # "quail": Quail("quail", tokenizer=tok, split_name="validation"),
    # "race": Race("race", tokenizer=tok, dataset_config_name="middle", split_name="test"),
    # "sst": SST("sst", "default", split_name='test')
# }

if __name__ == "__main__":
    data = RottenTomatoes("rotten_tomatoes", split_name="test")
    texts = []
    labels = []

    # answer_choices = ["bad", "good"]
    # answer_choices = ["negative", "positive"]
    # answer_choices = ["negative", "positive"]

    for template_name in ['Reviewer Opinion bad good choices', 'Text Expressed Sentiment', 'Movie Expressed Sentiment']:
        prompt = data.prompts[template_name]
        texts = []
        labels = []
        for example in data.dataset:
            dic = data.process_example(prompt, example)
            if dic is None:
                continue
            texts.append(dic["text_proc"])
            labels.append(dic["label_proc"])

        df = pd.DataFrame.from_dict({"text": texts, "label": labels, "prompt_id": template_name})
        df.to_csv("data/rotten_tomatoes_" + template_name + ".csv", index=False)

    pass
