import json
import math
import random
from tqdm import tqdm
import spacy
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-8B")


def process_and_write(solutions, sampled_data, output_file, batch_size=32, n_process=1, chunk_size=500):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer", "parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000

    with open(output_file, "w", encoding="utf-8") as fout:
        total = len(solutions)
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            chunk_solutions = solutions[start:end]
            chunk_sampled = sampled_data[start:end]

            pipe = nlp.pipe(chunk_solutions, batch_size=batch_size, n_process=n_process)
            for ex, doc in tqdm(zip(chunk_sampled, pipe), total=len(chunk_solutions), desc=f"chunk {start}-{end}"):
                sentences = [sent.text.strip() for sent in doc.sents]
                step = max(1, int(math.sqrt(len(sentences))))
                step = random.randint(1, step)
                partial_solution = " ".join(sentences[:step])
                prompt = f"Please reason step by step, and put your final answer within \\boxed{{}}: {ex['question']}"
                solution = partial_solution
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "flag"}]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompt_text = prompt_text.split("flag<|im_end|>")[0] + solution
                fout.write(
                    json.dumps(
                        {
                            "prompt": prompt_text,
                            "label": ex.get("final_answer"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        fout.flush()

if __name__ == "__main__":
    # 加载数据（保持你的版本）
    train_data = load_dataset("zwhe99/DeepMath-103K", split="train", cache_dir="./data/DeepMath-103K")
    assert isinstance(train_data, Dataset)
    sample_size = 5000
    indices = random.sample(range(len(train_data)), sample_size)
    sampled_data = [train_data[i] for i in indices]
    solutions = [ex["r1_solution_1"] for ex in sampled_data]

    output_file = "./data/deepmath_train_spacy.jsonl"

    # 配置建议：先用 n_process=1 或 2 测试，chunk_size=500 可视内存再调大或调小
    process_and_write(solutions, sampled_data, output_file, batch_size=32, n_process=1, chunk_size=500)
    print("Saved to", output_file)
