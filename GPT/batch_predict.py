import argparse
import ast
import json
from typing import Dict

import torch

from GPT.utils import decode, encode, load_model_from_checkpoint


class LLM:  # pragma: no cover, saving results
    def __init__(self, checkpoint: str, config: Dict) -> None:
        """initialize the gpt model

        Args:
            checkpoint (str): path to the checkpoint to load the model from
            config (Dict): configuration dictonary for the model
        """
        self.gpt = load_model_from_checkpoint(checkpoint, config)

    def __call__(self, prompts):
        maxlen = len(max(prompts, key=lambda x: len(x)))
        for i, prompt in enumerate(prompts):
            prompts[i] = prompts[i] + " " * (maxlen - len(prompt))

        prompts_encoded = [encode(prompt) for prompt in prompts]
        inputs = torch.tensor(prompts_encoded, dtype=torch.long)
        generations = self.gpt.generate(inputs)
        generations = generations.tolist()
        generations = [decode(generation) for generation in generations]
        return generations


if __name__ == "__main__":  # pragma: no cover, saving results
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", help="path to the experiment in from which the model should come")
    parser.add_argument("--input_batch", help="list of input strings to be completed via the LLM")
    args = parser.parse_args()
    input_batch = ast.literal_eval(args.input_batch)
    with open(f"../shared_storage/results/{args.experiment_name}", "r", encoding="utf-8") as f:
        checkpoint_metadata = json.load(f)

    checkpoint_metadata.pop("experiment_name")
    checkpoint_path = checkpoint_metadata.pop("best_tunning_checkpoint_dir")
    completer = LLM(checkpoint=checkpoint_path, config=checkpoint_metadata)

    print(completer(input_batch))
