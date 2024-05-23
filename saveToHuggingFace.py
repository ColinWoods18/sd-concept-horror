#@title Save your newly created concept to the [library of concepts](https://huggingface.co/sd-concepts-library)?

import os
import requests
from slugify import slugify
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from huggingface_hub import create_repo
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel

#just copy pasted to make sure everything worksimport argparse
import itertools
import math
import os
import random
import requests
from io import BytesIO


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import requests
import glob

import PIL
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



save_concept_to_public_library = True #@param {type:"boolean"}
name_of_your_concept = "horror-spooky" #@param {type:"string"}
#@markdown `hf_token_write`: leave blank if you logged in with a token with `write access` in the [Initial Setup](#scrollTo=KbzZ9xe6dWwf). If not, [go to your tokens settings and create a write access token](https://huggingface.co/settings/tokens)
hf_token_write = "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" #@param {type:"string"}
#@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
placeholder_token = "<horror-spooky>" #@param {type:"string"}
what_to_teach = "style" #@param ["style", "concept"]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2" #@param ["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-base", "CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"] {allow-input: true}
save_path = "Horror"

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 2000,
    "save_steps": 250,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": "sd-concept-output"
}


if save_concept_to_public_library:
    repo_id = f"sd-concepts-library/{slugify(name_of_your_concept)}"
    output_dir = hyperparameters["output_dir"]
    if not hf_token_write:
        with open(HfFolder.path_token, 'r') as fin: 
            hf_token = fin.read()
    else:
        hf_token = hf_token_write

    # Join the Concepts Library organization if you aren't part of it already
    headers = {
        'Authorization': f'Bearer {hf_token}',
        'Content-Type': 'application/json'
    }
    url = 'https://huggingface.co/organizations/sd-concepts-library/share/VcLXJtzwwxnHYCkNMLpSJCdnNFZHQwWywv'
    response = requests.post(url, headers=headers)

    images_upload = os.listdir("Horror")
    image_string = ""
    repo_id = f"sd-concepts-library/{slugify(name_of_your_concept)}"
    for i, image in enumerate(images_upload):
        image_string = f'''{image_string}![{placeholder_token} {i}](https://huggingface.co/{repo_id}/resolve/main/concept_images/{image})
    '''
    if(what_to_teach == "style"):
        what_to_teach_article = f"a `{what_to_teach}`"
    else:
        what_to_teach_article = f"an `{what_to_teach}`"
    readme_text = f'''---
license: mit
base_model: {pretrained_model_name_or_path}
---
### {name_of_your_concept} on Stable Diffusion
This is the `{placeholder_token}` concept taught to Stable Diffusion via Textual Inversion. You can load this concept into the [Stable Conceptualizer](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb) notebook. You can also train your own concepts and load them into the concept libraries using [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb).

Here is the new concept you will be able to use as {what_to_teach_article}:
    {image_string}
    '''
    #Save the readme to a file
    readme_file = open("README.md", "w")
    readme_file.write(readme_text)
    readme_file.close()
    #Save the token identifier to a file
    text_file = open("token_identifier.txt", "w")
    text_file.write(placeholder_token)
    text_file.close()
    #Save the type of teached thing to a file
    type_file = open("type_of_concept.txt","w")
    type_file.write(what_to_teach)
    type_file.close()
    operations = [
        CommitOperationAdd(path_in_repo="learned_embeds.bin", path_or_fileobj=f"{output_dir}/learned_embeds.bin"),
        CommitOperationAdd(path_in_repo="token_identifier.txt", path_or_fileobj="token_identifier.txt"),
        CommitOperationAdd(path_in_repo="type_of_concept.txt", path_or_fileobj="type_of_concept.txt"),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="README.md"),
    ]
    api = HfApi()
    if repo_id not in [model.modelId for model in api.list_models(token=hf_token)]:
        create_repo(repo_id, private=True, token=hf_token)
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload the concept {name_of_your_concept} embeds and token",
            token=hf_token
        )
    api.upload_folder(
        folder_path=save_path,
        path_in_repo="concept_images",
        repo_id=repo_id,
        token=hf_token
    )
    
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

#@title Set up the pipeline 

pipe = StableDiffusionPipeline.from_pretrained(
hyperparameters["output_dir"],
scheduler=DPMSolverMultistepScheduler.from_pretrained(hyperparameters["output_dir"], subfolder="scheduler"),
torch_dtype=torch.float16,
).to("cuda")

#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt

prompt = "a <horror-spooky> in a dark forest" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=30, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_rows, num_samples)
grid.show()

