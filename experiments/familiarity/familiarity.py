import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openai
import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# 4.1 uses the same encoding as 4o
enc = tiktoken.encoding_for_model("gpt-4o")

# Paths relative to this file
competitions_dir = Path("../../timeseriesgym/competitions/")


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(retry_if_exception_type((openai.error.APIError, IndexError))),
)
def get_prob(prompt: str, target_token: str, logprobs: int = 10) -> float | None:
    """
    Get the probability of the target token given the prompt.

    If the token is not one of the `logprobs` most probable tokens, returns None.
    """
    engine = "gpt-4.1-2025-04-14"  # Fill in with real base model engine to use
    completion_prompt = """Complete this sentence. You are acting as auto-complete.
    Simply complete the sentence to the best of your ability,
    make sure it is just ONE sentence: {sentence}"""
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {
                "role": "user",
                "content": completion_prompt.format(sentence=prompt),
            }
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=logprobs,
    )
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs_dict = {}
    for response in top_logprobs:
        token = response["token"].strip()
        top_logprobs_dict[token] = response["logprob"]
    target_token = target_token.strip()
    if target_token in top_logprobs_dict:
        logprob = top_logprobs_dict[target_token]
        # convert logprob to probability
        return np.exp(logprob)
    return None


def get_familiarity_score(
    text: str,
    n_samples: int,
    max_tokens: int = 1000,
) -> float:
    """
    Familiarity score is the mean probability of the target token over `n_samples` random places
    in the text.

    Example:
    ```
    test_str = "The quick brown fox jumps over the lazy dog"
    print(get_familiarity_score(test_str, n))
    test_str = "The soft-spoken purple invisible man leapt over the carton of pomegrenate juice"
    print(get_familiarity_score(test_str, n))
    ```
    """
    tokens = enc.encode(text)
    n_samples = min(n_samples, len(tokens))
    token_idxs = np.random.choice(len(tokens), n_samples, replace=False)

    total_prob = 0
    for i in tqdm(token_idxs):
        start = max(0, i - max_tokens)
        prompt_str = enc.decode(tokens[start:i])
        completion_gt = enc.decode([tokens[i]])
        prob = get_prob(prompt_str, completion_gt)
        if prob is not None:
            total_prob += prob
    return total_prob / n_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--relevant-doc-file",
        type=str,
        required=False,
        help="File with relevant documents to test familiarity",
    )
    parser.add_argument(
        "--output-cache-file",
        type=str,
        default="familiarity.json",
        help="Cache the results to a file",
    )
    parser.add_argument(
        "--output-plot-file",
        type=str,
        default="familiarity_results.pdf",
        help="Output plot file",
    )
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the model")
    parser.add_argument(
        "--n-completions", type=int, default=1, help="Number of completions to generate"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Number of samples to test familiarity"
    )
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of workers for ThreadPoolExecutor",
    )
    parser.add_argument(
        "--existing-results",
        type=str,
        default=None,
        help="Use a cached results file, so we just plot the results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of context to include for next token prediction",
    )
    args = parser.parse_args()

    familiarity_and_dates = {}
    with open(args.relevant_doc_file) as file:
        relavant_doc_file = json.load(file)

    empty_docs = []
    for comp_id, relavant_docs in relavant_doc_file.items():
        for doc_id, doc in relavant_docs.items():
            if doc == "":
                empty_docs.append((comp_id, doc_id))

    for comp_id, doc_id in empty_docs:
        del relavant_doc_file[comp_id][doc_id]
        print(f"Removed empty document {doc_id} from {comp_id}")

    comps_list = list(relavant_doc_file.keys())
    print(f"Found {len(comps_list)} competitions to process")

    if args.existing_results is not None:
        with open(args.existing_results) as file:
            familiarity_and_dates = json.load(file)
        print(f"Reusing existing results from {args.existing_results}")

    else:
        # Loop through all "description.md" files in the competitions directory
        tasks = []
        for seed in range(args.n_seeds):
            for comp_id in comps_list:
                tasks.append((comp_id, seed))

        def process_competition(_args):
            comp_id, seed = _args
            print(f"Processing {comp_id} with seed {seed}")
            result = {}
            # Load the text from the file
            if "missing" in comp_id:
                parent_id = comp_id.split("-missing")[0]
                description_file = competitions_dir / parent_id / "description.md"
            elif "MIT-BIH-Arrhythmia-Weak-Supervision" in comp_id:
                description_file = competitions_dir / comp_id / "description_label_and_model.md"
            else:
                description_file = competitions_dir / comp_id / "description.md"

            with description_file.open("r", encoding="utf-8") as file:
                description_text = file.read()
            document_names = ["description.md"]
            discussions_texts = []
            for key, value in relavant_doc_file[comp_id].items():
                discussions_texts.append(value)
                document_names.append(key)
            documents = [
                description_text,
                *discussions_texts,
            ]
            result["familiarity_scores"] = {}
            for document_name, document_text in zip(document_names, documents, strict=False):
                score = get_familiarity_score(document_text, args.n_samples, args.max_tokens)
                result["familiarity_scores"][document_name] = score
            # mean familiarity score across documents
            result["mean_familiarity_score"] = sum(result["familiarity_scores"].values()) / len(
                result["familiarity_scores"]
            )

            return comp_id, seed, result

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(executor.map(process_competition, tasks))

        for comp_id, seed, result in results:
            familiarity_and_dates[f"{comp_id}_seed{seed}"] = result

        # save the scores to file
        with open(args.output_cache_file, "w") as file:
            json.dump(familiarity_and_dates, file, indent=4, sort_keys=True)
