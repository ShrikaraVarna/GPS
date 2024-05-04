from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistral-trained-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def get_model_output(prompt):

    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=50,
        num_return_sequences=1,
    )
    return sequences[0]['generated_text']

from datasets import load_dataset
dataset = load_dataset("AI4Math/MathVerse", "testmini")
print(dataset)
print(len(dataset["testmini"]))

output_file = open("output.txt", "w")
for i in range(len(dataset['testmini'])):
  if dataset['testmini'][i]['problem_version'] == 'Text Dominant':
    sample = dataset['testmini'][i]
    type = sample['question_type']
    question = sample['question']
    answer = sample['answer']
    prompt = "###Instruction: First perform reasoning based on the information given in the question, then finally select the answer from the choices in the following format: Answer: xxx.\n"
    prompt += '###Question: ' + question + "\n###Answer:"
    generated_answer = get_model_output(prompt)
    output_file.write(str({'generated_answer': generated_answer, 'answer': answer, 'question': question, 'type': type}) + "\n")
