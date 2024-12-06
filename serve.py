
from fastapi import FastAPI
from ray import serve
import requests
from starlette.requests import Request
from utils import load_model_from_checkpoint, encode, decode
from config import logger
import torch
import argparse
import json
import ray
from typing import Dict



app = FastAPI(
    title="gpt completion",
    description="Tries to produce shakespeare off of your input",
    version="0.1")


@serve.deployment(num_replicas="1", ray_actor_options={"num_cpus": 8, "num_gpus": 0})
@serve.ingress(app)
class servedGPT:
    def __init__(self, checkpoint: str, config: Dict) -> None:
        """initialize the gpt model

        Args:
            checkpoint (str): path to the checkpoint to load the model from
            config (Dict): configuration dictonary for the model
        """
        self.checkpoint = checkpoint
        self.model = load_model_from_checkpoint(checkpoint, config)
         
    @app.get("/")
    async def _sayHi(self, request: Request):
        logger.info("Your LLM application started")
        return "200, ok!"
        
    @app.post("/predict/")
    async def _predict(self, request: Request):
        data = await request.json()
        prompt = data["prompt"]
        encoded_prompt = encode(prompt)
        input_tensor = torch.tensor([encoded_prompt], dtype=torch.long) # shape 1*T
        generation = self.model.generate(input_tensor)
        return decode(generation[0].tolist())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help="Name of the experiment to load from")
    args = parser.parse_args()
    with open(f"./results/{args.experiment_name}", 'r', encoding='utf-8') as f:
        checkpoint_metadata = json.load(f)

    checkpoint_metadata.pop('experiment_name')
    checkpoint_path = checkpoint_metadata.pop('best_tunning_checkpoint_dir')
    model_config = checkpoint_metadata
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    serve.run(servedGPT.bind(args.checkpoint_path, model_config))
