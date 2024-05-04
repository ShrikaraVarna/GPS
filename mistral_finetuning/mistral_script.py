from collections import defaultdict

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

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="<ADD TOKEN>")
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token="<ADD TOKEN>"
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
        #temperature=0.7,
        #top_k=50,
        #top_p=0.95,
        num_return_sequences=1,
    )
    return sequences[0]['generated_text']


demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. DO NOT provide an explanation, or anything extra. Please provide only 1 or 0 denoting if they are consistent or different respectively.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """

demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""

def create_test_prompt_for_extraction(demo_prompt, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt
def create_test_prompt(demo_prompt, question, answer, extraction):
    demo_prompt = demo_prompt.strip()
    full_prompt = demo_prompt.format(question = question, gt=answer, extraction=extraction)
    return full_prompt

def evaluate(question, ground_truth, model_gen):
    extracted_answer_prompt = create_test_prompt_for_extraction(demo_prompt_extract, model_gen)
    extraction_response = get_model_output(extracted_answer_prompt).strip()
    extracted_answer = extraction_response[extraction_response.rindex("Extracted Answer:") + len("Extracted Answer:"):].strip()
    full_prompt = create_test_prompt(demo_prompt_score, question, ground_truth, extracted_answer)
    response = get_model_output(full_prompt)
    response = response.strip()
    index = response.rindex('Judgement:')
    return int(response[index:].strip().split('\n')[0][len("Judgement:")+1:][0])

import json
lines = open("generated_answers.json", "r").readlines()
accuracy = 0
type_wise_scores = {"multi-choice": (0, 0), "free-form": (0, 0)}
for line in lines:
    object = json.loads(line)
    question = object["question"]
    answer = object["answer"]
    generated_answer = object['generated_answer']
    type = object['type']
    judgement = evaluate(question, generated_answer, answer)
    accuracy += judgement
    type_wise_scores[type][0] += judgement
    type_wise_scores[type][1] += 1
print("Accuracy: ", accuracy/len(lines))
print("Type wise scores: ", type_wise_scores)
