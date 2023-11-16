from transformers import T5ForConditionalGeneration, AutoTokenizer
import json
from datasets import load_dataset
from tqdm import tqdm
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = T5ForConditionalGeneration.from_pretrained("/home/dungbt/distilling-step-by-step/tmp/code-llama-output")
tokenizer = AutoTokenizer.from_pretrained("/home/dungbt/distilling-step-by-step/tmp/code-llama-output")
model.to(device)

model.eval()

def read_contextual_medit_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            examples.append(js['input'])
    return examples

def write_string_to_file(absolute_filename, string):
    with open(absolute_filename, 'a') as fout:
        fout.write(string)

examples = read_contextual_medit_examples('/home/dungbt/distilling-step-by-step/datasets/cmg/cmg_test.json')

predictions = []

for text in tqdm(examples):
    input_ids = tokenizer('predict: ' + text, max_length=200, truncation=True, return_tensors="pt").to(device).input_ids
    outputs = model.generate(input_ids)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    write_string_to_file('generated_predictions.txt', '' + prediction + '\n')
# model_inputs = tokenizer(['predict: ' + text for text in examples], padding=True, max_length=200, truncation=True, return_tensors="pt")

# outputs = model.generate(
#     input_ids=model_inputs["input_ids"],
#     attention_mask=model_inputs["attention_mask"],
#     max_new_tokens=32, 
#     do_sample=False, # disable sampling to test if batching affects output
# )   

# predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# output_prediction_file = "generated_predictions.txt"
# with open(output_prediction_file, "w", encoding="utf-8") as writer:
#     writer.write("\n".join(predictions))