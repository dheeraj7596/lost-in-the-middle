import sys

sys.path.append('/data/dmekala/lost-in-the-middle/src')
import argparse
import json
import logging
import pathlib
from copy import deepcopy
from functools import partial
from utils import modified_model_forward, modified_layer_forward
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

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


def main(
        model_name,
        alpha,
        layer_threshold,
        bsize,
        debug,
        temperature,
        top_p,
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
    did_format_warn = False

    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
    test_dataset = dataset["eval"]

    # Fetch all of the prompts
    for input_example in tqdm(test_dataset):
        instruction = input_example["instruction"]
        prompt = get_prompt(instruction)

        if "chat" in model_name:
            if did_format_warn is False:
                logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                did_format_warn = True
            prompt = format_chat_prompt(prompt)

        # prompt_length = len(tokenizer(prompt)["input_ids"])
        prompt_length = len(tokenizer.encode(prompt))
        if max_prompt_length < prompt_length:
            logger.info(
                f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                f"is greater than maximum prompt length {max_prompt_length}"
            )
            continue

        prompts.append(prompt)
        examples.append(deepcopy(input_example))

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
    raw_responses = model.inference(prompts, generation_config)
    idx = 0
    responses = []
    for p, s in zip(prompts, raw_responses):
        print(idx)
        ans = s.replace(model.tokenizer.eos_token, "").replace("<s>", "").strip().split(p.replace("<s>", "").strip())[
            -1].strip()
        print(p)
        print("Final Pred:", ans)
        print("*" * 80)
        idx += 1
        responses.append(ans)

    output_json = []
    for i, sample in enumerate(test_dataset):
        ans = responses[i]
        sample["output"] = ans
        sample["generator"] = model_name
        output_json.append(sample)

    with open(output_path, "w") as outfile:
        outfile.write(json.dumps(output_json, indent=4))


def get_prompt(text):
    return f"""

{text}

"""


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
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
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
    print("SEED HAS BEEN SET TO:", args.seed)
    main(
        args.model,
        args.alpha,
        args.layer_threshold,
        args.bsize,
        args.debug,
        args.temperature,
        args.top_p,
        args.max_new_tokens,
        args.max_prompt_length,
        args.output_path,
    )