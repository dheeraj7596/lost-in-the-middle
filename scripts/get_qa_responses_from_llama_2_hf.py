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
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from xopen import xopen

sys.path.append('/data/dmekala/lost-in-the-middle/src')
from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def chunks_by_size(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class ModelWrapper:
    def __init__(self, checkpoint, tokenizer_name=None, gpu_batch_size=16):
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


def main(
        input_path,
        model_name,
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
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            if "chat" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
                prompt = format_chat_prompt(prompt)

            prompt_length = len(tokenizer(prompt)["input_ids"])
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    logger.info("Loading model")
    model = ModelWrapper(model_name, gpu_batch_size=64)
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
    raw_responses = model.inference(prompts, generation_config)
    idx = 0
    responses = []
    for p, s in zip(prompts, raw_responses):
        print(idx)
        ans = s.replace(model.tokenizer.eos_token, "").strip().split(p)[-1].strip()
        print("Final Pred:", ans)
        print("*" * 80)
        responses.append(ans)

    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


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
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
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
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=4096,
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
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
