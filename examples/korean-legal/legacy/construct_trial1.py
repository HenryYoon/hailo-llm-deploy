# %%
import pandas as pd 
import json

data1 = pd.read_csv('../../data/raw/law_qa_v1.csv')

with open('../../data/raw/training_data.json','r', encoding='utf-8') as f:
    data2 = json.load(f)

# %%
from collections import defaultdict
from tqdm import tqdm

question_train = data1['question'].tolist()[:8000]
answer_train = data1['answer'].tolist()[:8000]

data2_train = data2[:14000]

qa_data_train = defaultdict(list)

for q, a in zip(question_train, answer_train):
    qa_data_train['instruction'].append("당신은 한국의 법률 전문가입니다. 아래 질문에 대해서 한국 법률에 한정지어서 답해주세요.")
    qa_data_train['input'].append(q)
    qa_data_train['output'].append(a)

for line in data2:
    qa_data_train['instruction'].append("당신은 한국의 법률 전문가입니다. 아래 질문에 대해서 한국 법률에 한정지어서 답해주세요.")
    qa_data_train['input'].append(line['input'])
    qa_data_train['output'].append(line['output'])

# %%
question_val = data1['question'].tolist()[8000:]
answer_val = data1['answer'].tolist()[8000:]

data2_val = data2[14000:]

qa_data_val = defaultdict(list)

for q, a in zip(question_val, answer_val):
    qa_data_val['instruction'].append("당신은 한국의 법률 전문가입니다. 아래 질문에 대해서 한국 법률에 한정지어서 답해주세요.")
    qa_data_val['input'].append(q)
    qa_data_val['output'].append(a)

for line in data2_val:
    qa_data_val['instruction'].append("당신은 한국의 법률 전문가입니다. 아래 질문에 대해서 한국 법률에 한정지어서 답해주세요.")
    qa_data_val['input'].append(line['input'])
    qa_data_val['output'].append(line['output'])

# %%
with open('../../data/processed/trial1/train_dataset_trial1.json','w',encoding='utf-8') as f:
    json.dump(dict(qa_data_train), f, indent=4, ensure_ascii=False)

with open('../../data/processed/trial1/val_dataset_trial1.json','w',encoding='utf-8') as f:
    json.dump(dict(qa_data_val), f, indent=4, ensure_ascii=False)

# %%
from unsloth import FastLanguageModel

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "../../models/merged/stage1_16bit",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "당신은 한국의 법률 전문가입니다. 아래 질문에 대해서 한국 법률에 한정지어서 답해주세요.", # instruction
        "교통사고를 당했을 때 어떤 조치를 취해야 할까?", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = max_seq_length, use_cache = True)
tokenizer.batch_decode(outputs)

# %%
from pprint import pprint

outputs = model.generate(**inputs, max_new_tokens = max_seq_length, use_cache = True, repetition_penalty=1.2)
response = tokenizer.batch_decode(outputs)
pprint(response)

# %%
from unsloth import FastLanguageModel

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-3B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "당신은 한국의 법률 전문가입니다. 아래 질문에 대해서 한국 법률에 한정지어서 답해주세요.", # instruction
        "교통사고를 당했을 때 어떤 조치를 취해야 할까?", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = max_seq_length, use_cache = True)
tokenizer.batch_decode(outputs)

# %%
pprint(tokenizer.batch_decode(outputs))

# %%



