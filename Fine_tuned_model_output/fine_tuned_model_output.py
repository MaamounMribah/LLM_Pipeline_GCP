import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

coqa = pd.read_json('QA.json')
del coqa["version"]

#required columns in our dataframe
cols = ["text","question","answer"]
#list of lists to create our dataframe
comp_list = []
for index, row in coqa.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = []
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["input_text"])
        temp_list.append(row["data"]["answers"][i]["input_text"])
        comp_list.append(temp_list)
new_df = pd.DataFrame(comp_list, columns=cols) 
#saving the dataframe to csv file for further loading
new_df.to_csv("CoQA_data.csv", index=False)

data = pd.read_csv("CoQA_data.csv")
print("Number of question and answers: ", len(data))

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

random_num = np.random.randint(0,len(data))
question = data["question"][random_num]
text = data["text"][random_num]

input_ids = tokenizer.encode(question, text)

tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token,id))

sep_idx = input_ids.index(tokenizer.sep_token_id)
num_seg_a = sep_idx+1

num_seg_b = len(input_ids) - num_seg_a

segment_ids = [0]*num_seg_a + [1]*num_seg_b
assert len(segment_ids) == len(input_ids)

#token input_ids to represent the input and token segment_ids to differentiate our segments - question and text
output = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))

#tokens with highest start and end scores
answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)
if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end+1])
else:
    print("I am unable to find the answer to this question. Can you please ask another question?")

answer = tokens[answer_start]
for i in range(answer_start+1, answer_end+1):
    if tokens[i][0:2] == "##":
        answer += tokens[i][2:]
    else:
        answer += " " + tokens[i]
    
print("\nQuestion:\n{}".format(question.capitalize()))
print("\nAnswer:\n{}.".format(answer.capitalize()))