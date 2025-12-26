from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500",split="test",cache_dir="./data/MATH-500")

def process_fn(example):
    return {
        "prompt": [{"role": "system", "content": (
            "You are a helpful assistant that solves math problems.\n"
            "Reason step by step.\n"
            "At the end, output the final answer in the format \\boxed{answer}."
        )},{"role": "user", "content": example["problem"]}],
        "label": example["answer"]
    }

ds = ds.map(process_fn, remove_columns=ds.column_names)
ds.to_json("data/math_test.jsonl")