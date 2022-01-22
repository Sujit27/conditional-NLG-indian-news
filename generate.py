import os
import io
import requests
import numpy as np
import pandas as pd
import re
import zipfile
import random
import time
import csv
import datetime
from itertools import compress
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

from IPython.display import clear_output

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'

print(f"Device available: {DEVICE}")

MODEL           = 'models' #{gpt2, gpt2-medium, gpt2-large, gpt2-xl}
SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
MAXLEN          = 500


def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained('gpt2') #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        # print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path,map_location=torch.device(DEVICE)))

    model.to(DEVICE)
    return model

def generate_text(title,keywords,model,tokenizer):
    kw = ','.join(keywords)

    prompt = SPECIAL_TOKENS['bos_token'] + title + \
            SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
            
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(DEVICE)

    model.eval()
    sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                min_length=50, 
                                max_length=MAXLEN,
                                top_k=30,                                 
                                top_p=0.7,        
                                temperature=0.9,
                                repetition_penalty=4.0,
                                early_stopping=True,
                                num_return_sequences=1
                                )

    return sample_outputs

def main():
    print("Loading text generator...")
    tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
    model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS,load_model_path=os.path.join(MODEL,'pytorch_model.bin'))



    while True:
        print("Type q/Q to quit")
        user_input_1 = input("Enter title/headline:") 
        if user_input_1 == 'q' or user_input_1 == 'Q':
            break

        user_input_2 = input("Enter some keywords separated by comma:")
        if user_input_2 == 'q' or user_input_1 == 'Q':
            break

        title = user_input_1
        keywords = user_input_2.split(",")


    # title = "Australia beats India by 7 wickets in Nagpur test"
    # keywords = ['Nagpur', 'Cricket', 'test', 'Kohli', 'win']
        print("Generating text...")
        output = generate_text(title,keywords,model,tokenizer)

        for i, sample_output in enumerate(output):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)
            a = len(title) + len(','.join(keywords))    
            print("{}: {}\n\n".format(i+1,  text[a:]))

if __name__ == "__main__":
    main()