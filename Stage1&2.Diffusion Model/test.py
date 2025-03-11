from __future__ import absolute_import, division, print_function
import sys
import os
import argparse
import json
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config
import einops
import random
import ast
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector



transform = transforms.Compose([           
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default="")
    
    parser.add_argument('--input_img', type=str, default="")

    parser.add_argument('--out_dir', type=str, default="")

    parser.add_argument('--strength', type=float, default=1.0)

    parser.add_argument('--depth_dir', type=str, default="")

    parser.add_argument('--eval', type=ast.literal_eval, default=False)
 
    parser.add_argument('--finetuned', type=ast.literal_eval, default=True)
    
    parser.add_argument('--weights', type=str, default="")

    parser.add_argument('--num_samples', type=int, default=1)  

    parser.add_argument('--prompt_file', type=str, default="")

    parser.add_argument('--prompt', type=str, default="")

    parser.add_argument('--n_iter', type=int, default=1)
  
    parser.add_argument('--adaptive_control', type=ast.literal_eval, default=False)

    parser.add_argument('--padding_bbox', type=int, default=0)
    
    # set seed
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    return args


args = parse_args()

model = create_model("./models/cldm_v15.yaml").cpu()
if args.finetuned:
    model.load_state_dict(load_state_dict(args.weights, location='cuda'), strict=False)
model = model.to("cuda")

gen_count = 1
    

with open(args.prompt_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
    
    for idx, item in enumerate(data_list):
        source_filename = item.get('source')
        print("This", source_filename)
        target_filename = item.get('target')
        prompt = item.get('prompt')

        
        image = cv2.imread(source_filename)
        layout_hint = cv2.imread(target_filename, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (512, 512))
        raw_image = image
        H, W, C = raw_image.shape

        image = (image.astype(np.float32) / 127.5) - 1.0 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        hint = layout_hint
        hint = cv2.resize(hint, (512,512))
        hint = hint.astype(np.float32) / 255.0
        
        for iteration in range(args.n_iter):
            
            ddim_sampler = DDIMSampler(model)
            num_samples = args.num_samples
            ddim_steps = 50  
            guess_mode = False
            strength = args.strength
            scale = 9.0
            seed = args.seed

            a_prompt = "realistic, best quality, extremely detailed"
            n_prompt = "fake 3D rendered image, bad anatomy, worst quality, low quality"

            source = raw_image
            source = cv2.resize(source, (512, 512))
            source = (source.astype(np.float32) / 127.5) - 1.0 
            source = source.transpose([2, 0, 1])  # source is c h w   #(3,512,512)
            
            hint = hint[
                None,
            ].repeat(3, axis=0) #(1,512,512)-->(3,512,512)
            
            hint = torch.stack(
                [torch.tensor(hint) for _ in range(num_samples)], dim=0
                ).to("cuda")

#-----------------------------------------------------------------------------------
            
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)


            cond = {
                "c_concat": [hint], 
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
                }
            un_cond = {
                "c_concat": None if guess_mode else [hint], 
                "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            if not args.adaptive_control:
                seed_everything(seed)
                model.control_scales = (
                    [strength * (0.825 ** float(12 - i)) for i in range(13)]
                    if guess_mode
                    else ([strength] * 13)
                )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                samples, intermediates = ddim_sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    # eta=eta,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=un_cond,
                )
            
                if config.save_memory:
                    model.low_vram_shift(is_diffusing=False)

                x_samples = model.decode_first_stage(samples)

                x_samples = (
                    (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                
        for i in range(args.num_samples):
            cv2.imwrite(
                        os.path.join(args.out_dir, "{}.png".format(gen_count)), cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR)
                        )
            print("gen_count", gen_count)
            gen_count += 1