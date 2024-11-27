import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_audio,
    get_model_name_from_path,
)
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _load_json(f_):
    with open(f_, "r") as fin:
        return json.load(fin)


def iterate_eval_datasets():
    """Assumed model input format for prediction/inference:
    {
        "local_audio_path": pointer_to_audio,
        "instruction": text_input,
        ...
    }

    """

    cap_dir = ""  # path to the input for model prediction; captioning
    captioning_inference = {
        "task_name": _load_json(os.path.join(cap_dir, "captioning_task_json.json")),
    }
    # reasoning
    rea_dir = ""  # path to the input for model prediction; resoning
    reasoning_inference = {
        "task_name": _load_json(os.path.join(rea_dir, "reasoning_task_json.json")),
    }
    return captioning_inference, reasoning_inference


def shorten_lyrics(instruction, N=100):
    verses = instruction.split("\n")
    shorted = []
    tot_words = 0
    for v in verses:
        vs = v.split(" ")
        tot_words += len(vs)
        if tot_words < N:
            shorted.append(v)
        else:
            break
    ret = "\n".join(shorted)
    if ret[-1] != ".":
        ret = ret + "."
    return ret


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        is_audio_model=True,
        audio_ckpt=args.audio_ckpt,
        device=args.device,
    )
    model = model.to(torch.bfloat16).cuda()
    captioning_questions, reasoning_questions = iterate_eval_datasets()
    for task_collection in [captioning_questions, reasoning_questions]:
        for task_name, questions in task_collection.items():
            print(f"Working on this dataset: {task_name}")
            questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
            ans_file = open(args.answers_file, "w")
            for idx, line in tqdm(enumerate(questions)):
                gold_label = line["output"]
                qs = line["instruction"]
                qs = shorten_lyrics(qs, 400)
                cur_prompt = qs
                if model.config.mm_use_im_start_end:
                    qs = (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + qs
                    )
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )
                audio = {"local_audio_path": line["local_audio_path"]}
                audio_tensor = process_audio([audio], image_processor, model.config)[0]
                audio_tensor = audio_tensor.unsqueeze(0)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids.cuda(),
                        images=audio_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                        image_sizes=[(1024 * 3, 128)],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=256,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ].strip()
                ans_id = shortuuid.uuid()
                ans_file.write(
                    json.dumps(
                        {
                            "question_id": idx,
                            "prompt": cur_prompt,
                            "text": outputs,
                            "answer_id": ans_id,
                            "model_id": model_name,
                            "metadata": {},
                            "gold_label": gold_label,
                            "local_audio_path": line["local_audio_path"],
                        },
                    )
                    + "\n"
                )
                ans_file.flush()
            ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file-folder", type=str, default="./")
    parser.add_argument("--answers-file", type=str, default="myanswer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--audio-ckpt",
        type=str,
        default="vitb_finetuned.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    eval_model(args)
