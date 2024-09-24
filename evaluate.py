import os
import torch
import argparse
from transformers import AutoTokenizer
from sat.model import ChatGLM3Model
from sat.model.mixins import CachedAutoregressiveMixin
from prompt import CRITIC_PROMPT_ZHV2, category_CN
import jsonlines
import re
from tqdm import tqdm
from torch.utils.data import Dataset
    

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--critic_model_path',
        type=str,
        default="/share/home/wuyuhang/working_dir/vlbackbone/checkpoints/critic6b",
        help="path to the critic model file. must in SAT format"
    )
    
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default="THUDM/chatglm3-6b",
        help="path to the model tokenizer file."
    )
    
    parser.add_argument(
        '--response_file',
        type=str,
        default="/share/home/wuyuhang/working_dir/alignmmbench/responses/YiVL_34B.jsonl",
        help="path to the model response file, should in .jsonl format"
    )
    
    parser.add_argument(
        '--metadata_file',
        type=str,
        default='data/metadata.jsonl',
        help="path to the metadata result file, should in .jsonl format"
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default='results/result.jsonl',
        help="path to save detailed evaluation results, should in .jsonl format"
    )
    
    args = parser.parse_args()
    return args

def parse_score(id, pred):
    pattern = r'"Rating":\s*(\d+),\s*"Reason":\s*(.*)\s*'
    match = re.search(pattern, pred)
    if match:
        return {"question_id": id, "Rating": int(match.group(1).strip()), "Reason": match.group(2).strip()}
    else:
        return {"question_id": id, "Rating": -1, "Reason": "Error"}

class ChatGLMEvaluator:
    def __init__(self, args):

        print(f"Building critic model from {args.critic_model_path} on device {device}, local rank {args.local_rank}, rank {args.rank}, world_size {args.world_size}.")
        model, _ =  ChatGLM3Model.from_pretrained(
                            args.critic_model_path,
                            args=argparse.Namespace(
                            deepspeed=None,
                            local_rank=args.local_rank,
                            rank=args.rank,
                            world_size=args.world_size,
                            model_parallel_size=1,
                            mode='inference',
                            skip_init=True,
                            use_gpu_initialization=True,
                            device=device
                        ), url='local')
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        model.eval()

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.args = argparse.Namespace(
            max_length=4096,
            temperature=0.8,
            top_p=0.4,
            top_k=1
        )
        self.sep = "<|assistant|>"
        
        print(f"Loading metadata from {args.metadata_file}")
        with jsonlines.open(args.metadata_file, 'r') as fp: self.meta_data = {row['question_id']: row for row in fp}
        

    def chat(self, tokens, **kwargs):
        from sat.generation.sampling_strategies import BaseStrategy
        from sat.generation.autoregressive_sampling import filling_sequence

        context_len = len(tokens)
        inputs = tokens.to(self.model.parameters().__next__().device)[0]
        seq = torch.cat(
            [inputs, torch.tensor([-1] * (self.args.max_length - len(inputs)), device=inputs.device)], dim=0
        )
        strategy = BaseStrategy(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            end_tokens=[self.tokenizer.eos_token_id]
        )
        output = filling_sequence(
            self.model, seq,
            batch_size=1,
            strategy=strategy,
            **kwargs
        )[0]  # drop memory
        output = output[0][context_len:].unsqueeze(0)
        pred = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        return pred

    def generate(self, prompt, history=[]):
        assert len(history) == 0, "history is not supported yet"
        # build conversations
        conversations = []
        for i, (old_query, response) in enumerate(history):
            conversations.append({"role": "user", "content": old_query})
            conversations.append({"role": "assistant", "content": response})
        conversations.append({"role": "user", "content": prompt})
        conversations.append({"role": "assistant", "content": ""})
        # format conversations
        input_ids = [self.tokenizer.get_command("[gMASK]"), self.tokenizer.get_command("sop")]
        for conv in conversations:
            text = self.tokenizer.build_single_message(conv['role'], "", conv["content"])
            input_ids.extend(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        # inference
        pred = self.chat(input_ids)
        pred = pred.split(self.sep)[-1].strip()
        return pred

    @torch.no_grad()
    def single_evaluate(self, response: dict) -> dict:
        # response: {"question_id": "", "predict": ""}
        def process_history(history):
            if not history:
                return ""
            # print(history)
            lst = ['       - **对话历史：**\n']
            for idx, messages in enumerate(history):
                lst.append(f"        【第 {idx+1} 轮】\n\n")
                lst.append(f"        问: {messages['user']}\n\n")
                lst.append(f"        答: {messages['assistant']}\n\n")
            return ''.join(lst)
        id, preds = response['question_id'], response['predict']
        data = self.meta_data[id]
        query = CRITIC_PROMPT_ZHV2["General"].format(
            question=data["prompt"],
            questionType=data["task"],
            refAnswer=data["ref_answer"],
            assistant=preds,
            warnings=CRITIC_PROMPT_ZHV2["Types"][category_CN[data["task"]]],
            history=process_history(data["history"])
            ).strip()
        
        return parse_score(id, self.generate(query))

args = parse_args()
args.__setattr__("local_rank", os.getenv("LOCAL_RANK", 0))
args.__setattr__("rank", os.getenv("RANK", 0))
args.__setattr__("world_size", os.getenv("WORLD_SIZE", 0))

glm_evaluator = ChatGLMEvaluator(args)

def evaluate(responses):
    results = []
    dst_path = args.save_path
    
    print("Evaluating......")
    for row in tqdm(responses):
        tmp = glm_evaluator.single_evaluate(row)
        with jsonlines.open(dst_path, mode='a') as fp:
            fp.write(tmp)
        results.append(tmp)
        
    scores = {}
    good_cnt = {}
    bad_cnt = 0
    for row in results:
        task_type = glm_evaluator.meta_data[row['question_id']]['task']
        if row['Rating'] == -1:
            bad_cnt += 1
        else:
            scores[task_type] = scores.get(task_type, 0) + row['Rating']
            good_cnt[task_type] = good_cnt.get(task_type, 0) + 1
    print(f"Bad cnt: {bad_cnt}")
    print(f"Total score: {sum(scores.values()) / sum(good_cnt.values())}")
    for category_en, score in scores.items():
        print(f"Task {category_en} score: {score / good_cnt[category_en]}")

if __name__ == "__main__":
    with jsonlines.open(args.response_file, 'r') as fp:
        responses = list(fp)
    evaluate(responses)