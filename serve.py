
from fastapi import FastAPI
from ray import serve
import requests
import logging
from starlette.requests import Request
from utils import load_model_from_checkpoint, encode, decode
import torch
import argparse
import json
import ray

logger = logging.getLogger("ray.serve")



app = FastAPI(
    title="gpt completion",
    description="Tries to produce shakespeare off of your input",
    version="0.1")


@serve.deployment(num_replicas="1", ray_actor_options={"num_cpus": 8, "num_gpus": 0})
@serve.ingress(app)
class servedGPT:
    def __init__(self, checkpoint, config):
        self.checkpoint = checkpoint
        self.model = load_model_from_checkpoint(checkpoint, config)
         
    @app.get("/")
    async def _sayHi(self, request: Request):
        logging.info("Your LLM application started")
        return "200, ok!"
        
    @app.post("/predict/")
    async def _predict(self, request: Request):
        logger.warning(f"this is your stupid {request}")
        data = await request.json()
        logger.info(f"your request data looks like this: {data}")
        prompt = data["prompt"]
        logger.info(f"your prompt is: {prompt}")
        encoded_prompt = encode(prompt)
        input_tensor = torch.tensor([encoded_prompt], dtype=torch.long) # shape 1*T
        generation = self.model.generate(input_tensor)
        return decode(generation[0].tolist())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help="Address for the checkpoint of the target model")
    parser.add_argument('model_config', help="configuration of the gpt model to be served")
    args = parser.parse_args()
    model_config = json.loads(args.model_config)
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    serve.run(servedGPT.bind(args.checkpoint_path, model_config))
