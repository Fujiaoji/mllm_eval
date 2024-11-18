import torch
import time
import os
import json
import pandas as pd
import multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "3" # set the visible gpus

def process_data(gpu_id, tag):
    print(f"Process has PID: {os.getpid()}")
    # GPU
    torch.cuda.set_device(gpu_id)    
    print(f"Process on GPU {gpu_id}")
    # Load model
    model_id = "meta-llama/XXXXXXXXX"

    model = MllamaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
        cache_dir="XXX/Llama/cache/"
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir="XXXX/Llama/cache/")
    # load prompt
    with open("jp_prompt.txt", "r") as f:
        question = f.read()

    pred_brand_llama(model, processor, question, gpu_id, tag)
    
    



def pred_brand_llama(model, scr_type, processor, input_question, gpu_id, tag):
    responce_info = []
    
    attempts = 0  # Initialize attempts counter
    while attempts < 3:
        try:
            with open("jp.txt", "r") as f:
                input_content = f.read()
            input_question_html = input_question + "\n" + input_content
            messages = [{"role": "user", "content": [{"type": "text", "text": input_question_html}]}, {"role": "assistant", "content": ""}]
            input_text = processor.apply_chat_template(messages, format = "json", add_generation_prompt=True, return_full_text=False)
            inputs = processor(images=None, text=input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

            output = model.generate(**inputs, max_new_tokens=100)

            answer = processor.decode(output[0])
            responce_info.append(answer) 
            break
        except:
            attempts += 1
            time.sleep(20)
            if attempts >= 3:
                break
    print(responce_info)

def main():
    process_data(0)

if __name__ == "__main__":
    main()
    


