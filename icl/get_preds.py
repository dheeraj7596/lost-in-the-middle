#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run Llama-2 to get responses.

Currently supports:

- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Llama-2-70b-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-chat-hf

The retrieval results are used in the exact order that they're given.
"""
import argparse
import json
import logging
import pathlib
import sys

from sklearn.metrics import accuracy_score

sys.path.append('/data/dmekala/lost-in-the-middle')
from copy import deepcopy
from functools import partial
from datasets import load_dataset

import pandas as pd
from scripts.utils import modified_model_forward, modified_layer_forward

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from xopen import xopen

sys.path.append('/data/dmekala/lost-in-the-middle/src')

logger = logging.getLogger(__name__)


def chunks_by_size(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class ModelWrapper:
    def __init__(self, checkpoint, alpha, layer_threshold, tokenizer_name=None, gpu_batch_size=16):
        self.gpu_batch_size = gpu_batch_size
        if tokenizer_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, padding_side='left', truncation_side="right")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, padding_side='left', truncation_side="right")
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                          torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True,
                                                          trust_remote_code=True,
                                                          attn_implementation="eager",
                                                          device_map="auto")
        self.alpha = alpha
        self.layer_threshold = layer_threshold
        self.reweight_attn()
        self.model.eval()

    def inference(self, prompts, generation_config, skip_special_tokens=False):
        all_outputs = []
        chunks = chunks_by_size(prompts, self.gpu_batch_size)
        # print(f"Making {len(chunks)} chunk(s) each with size {len(chunks[0])}")
        for batch_prompts in chunks:
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", add_special_tokens=False,
                                    padding=True, truncation=True, max_length=4096).to("cuda")
            print("inputs shape:", inputs.input_ids.shape)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **generation_config)
                all_outputs += self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=skip_special_tokens)
        return all_outputs

    def reweight_attn(self):
        self.model.model.forward = partial(modified_model_forward, self.model.model)
        for i, layer in enumerate(self.model.model.layers):
            layer.forward = partial(modified_layer_forward, layer, layer_threshold=self.layer_threshold,
                                    alpha=self.alpha)


def reformat(dataset_name, text=None, sent1=None, sent2=None, question=None):
    if dataset_name == "rotten_tomatoes":
        prompt_format = """<s>Text: compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .
Label: 1
###
Text: sunk by way too much indulgence of scene-chewing , teeth-gnashing actorliness .
Label: 0
###
Text: the soundtrack alone is worth the price of admission .
Label: 1
###
Text: what the audience feels is exhaustion , from watching a movie that is dark ( dark green , to be exact ) , sour , bloody and mean .
Label: 0
###
Text: {text}
Label:"""
        return prompt_format.format_map({"text": text})
    elif dataset_name == "ag_news":
        prompt_format = """<s>Text: Talks End With No U.S. Climate Deal A U.N. conference ended early Saturday with a vague plan for informal new talks on how to slow global warming but without a U.S. commitment to multilateral negotiations on next steps, including emissions controls.
Label: 0
###
Text: Texas' Johnson, Benson Go Out With Win (AP) AP - Their final games will be remembered for the plays others made. Still, Texas tailback Cedric Benson and linebacker Derrick Johnson went out the way they wanted to: with a Rose Bowl win.
Label: 1
###
Text: Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.
Label: 2
###
Text: Thunderbird well worth a test flight Many people don #39;t pay all that much attention to their e-mail software. After all, it takes a real geek to care about the fine points of one program or another, especially when they all do more or less the same thing.
Label: 3
###
Text: Scant Progress on Post-Kyoto as Climate Talks End (Reuters) Reuters - U.N. talks on climate change ended early Saturday with few steps forward as the United States, oil producers and developing giants slammed the\brakes on the European Union's drive for deeper emissions cuts to stop global warming.
Label: 0
###
Text: Surprise! Ratner #39;s team can play to win It remains to be seen whether the Nets are as good with Vince Carter as they once were with Kenyon Martin, but at least they have made winning a priority again over in the Meadowlands.
Label: 1
###
Text: Murdoch offers $44 million for Manhattan penthouse MUMBAI: Media moghul and billionaire chairman of News Corp Rupert Murdoch has, according to agency reports, offered to buy a penthouse in New York #39;s upscale Manhattan area for cool $44 million.
Label: 2
###
Text: OSDL Looks Under the Sofa Cushions for Signs of Linux Growth The Open Source Development Labs has gone into the soothsayer business and - based on research that it had IDC run up - says that the global Linux market will be worth $35.7 billion in 2008.
Label: 3
###
Text: {text}
Label:"""
        return prompt_format.format_map({"text": text})
    elif dataset_name == "rte":
        prompt_format = """<s>Sentence1: No Weapons of Mass Destruction Found in Iraq Yet.
Sentence2: Weapons of Mass Destruction Found in Iraq.
Label: 1
###
Sentence1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.
Sentence2: Pope Benedict XVI is the new leader of the Roman Catholic Church.
Label: 0
###
Sentence1: Brian Brohm, the Louisville quarterback, threw for 368 yards and five touchdowns as the Cardinals beat visiting Oregon State 63-27.
Sentence2: The quarterback threw for 413 yards and three touchdowns, and then ran to the end zone two more times.
Label: 1
###
Sentence1: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients.
Sentence2: Herceptin can be used to treat breast cancer.
Label: 0
###
Sentence1: {sent1}
Sentence2: {sent2}
Label:"""
        return prompt_format.format_map({"sent1": sent1, "sent2": sent2})
    elif dataset_name == "gsm8k":
        prompt_format = f"""<s>Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
###
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10
###
Question: {question}
Answer:"""
        return prompt_format.format_map({"question": question})
    else:
        raise Exception("unknown dataset")


def post_process(dataset_name, ans):
    if dataset_name == "rotten_tomatoes" or dataset_name == "rte" or dataset_name == "ag_news":
        try:
            pred = int(ans.split("\n")[0].strip())
        except:
            pred = -1
    elif dataset_name == "gsm8k":
        try:
            pred = int(ans.split("####")[-1].strip())
        except:
            pred = -1
    else:
        raise Exception("unknown dataset")
    return pred


def main(
        dataset_name,
        model_name,
        alpha,
        layer_threshold,
        bsize,
        debug,
        temperature,
        top_p,
        closedbook,
        prompt_mention_random_ordering,
        use_random_ordering,
        query_aware_contextualization,
        max_new_tokens,
        max_prompt_length,
        output_path,
):
    global tasks
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    examples, gts, prompts = get_data(dataset_name, max_prompt_length, tokenizer)

    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    logger.info("Loading model")
    model = ModelWrapper(model_name, alpha=alpha, layer_threshold=layer_threshold, gpu_batch_size=bsize)
    if temperature != 0:
        do_sample = True
    else:
        do_sample = False
    generation_config = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
    }
    if debug:
        prompts = prompts[:100]
        examples = examples[:100]
        gts = gts[:100]
    raw_responses = model.inference(prompts, generation_config)
    idx = 0
    responses = []
    preds = []
    for p, s in zip(prompts, raw_responses):
        print(idx)
        ans = s.replace(model.tokenizer.eos_token, "").replace("<s>", "").strip().split(p.replace("<s>", "").strip())[
            -1]
        pred = post_process(dataset_name, ans)
        print("Final Pred:", pred)
        print("GT:", gts[idx])
        print("*" * 80)
        preds.append(pred)
        responses.append(ans)
        idx += 1

    print("Accuracy:", accuracy_score(gts, preds))

    with xopen(output_path, "w") as f:
        for example, prompt, response in zip(examples, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def get_data(dataset_name, max_prompt_length, tokenizer):
    examples = []
    prompts = []
    gts = []
    # Fetch all of the prompts
    if dataset_name == "rotten_tomatoes":
        df = load_dataset("rotten_tomatoes", split="test")
        for input_example in df:
            # Get the prediction for the input example
            text = input_example["text"]
            label = input_example["label"]
            prompt = reformat(dataset_name, text=text)
            # prompt_length = len(tokenizer(prompt)["input_ids"])
            prompt_length = len(tokenizer.encode(prompt))
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            prompts.append(prompt)
            examples.append(deepcopy(dict(input_example)))
            gts.append(label)
    elif dataset_name == "ag_news":
        df = load_dataset("ag_news", split="test")
        for input_example in df:
            # Get the prediction for the input example
            text = input_example["text"]
            label = input_example["label"]
            prompt = reformat(dataset_name, text=text)
            # prompt_length = len(tokenizer(prompt)["input_ids"])
            prompt_length = len(tokenizer.encode(prompt))
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            prompts.append(prompt)
            examples.append(deepcopy(dict(input_example)))
            gts.append(label)
    elif dataset_name == "rte":
        df = load_dataset("nyu-mll/glue", "rte", split="validation")
        for input_example in df:
            sent1 = input_example["sentence1"]
            sent2 = input_example["sentence2"]
            label = input_example["label"]
            prompt = reformat(dataset_name, sent1=sent1, sent2=sent2)
            # prompt_length = len(tokenizer(prompt)["input_ids"])
            prompt_length = len(tokenizer.encode(prompt))
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            prompts.append(prompt)
            examples.append(deepcopy(dict(input_example)))
            gts.append(label)
    elif dataset_name == "gsm8k":
        df = load_dataset("gsm8k", 'main', split="test")
        for input_example in df:
            question = input_example["question"]
            answer = input_example["answer"].split("####")[-1].strip()
            prompt = reformat(dataset_name, question=question)
            # prompt_length = len(tokenizer(prompt)["input_ids"])
            prompt_length = len(tokenizer.encode(prompt))
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            prompts.append(prompt)
            examples.append(deepcopy(dict(input_example)))
            gts.append(int(answer))
    else:
        raise Exception("unknown dataset")
    return examples, gts, prompts


def format_chat_prompt(message: str):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
    )
    lines = ["<s>[INST] <<SYS>>", DEFAULT_SYSTEM_PROMPT, "<</SYS>>", "", f"{message} [/INST]"]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", help="Name of the dataset", required=True)
    # parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, runs only on 100 samples",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha to weight residual",
        default=1.0
    )
    parser.add_argument(
        "--layer_threshold",
        type=int,
        help="Alpha to weight residual",
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed",
        default=42
    )
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--bsize",
        help="Batch size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=4096,
    )
    args = parser.parse_args()
    set_seed(args.seed)

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.dataset_name,
        args.model,
        args.alpha,
        args.layer_threshold,
        args.bsize,
        args.debug,
        args.temperature,
        args.top_p,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.max_prompt_length,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
