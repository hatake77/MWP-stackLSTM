import json

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

new_data = {}
sentences = []
equ_dict = {}
for elem in read_data_json("../../data/train23k_processed.json"):
    sentence = elem['text'].strip().split(' ')
    equation = elem['target_norm_post_template'][2:]  # .strip().split(' ')
    for equ_e in equation:
        if equ_e not in equ_dict:
            equ_dict[equ_e] = 1
    sentences.append(sentence)
    for elem in sentence:
        new_data[elem] = new_data.get(elem, 0) + 1
token_list = ['PAD_token', 'UNK_token', 'END_token']
ext_list = ['PAD_token', 'END_token']
for k, v in sorted(new_data.items()):
    token_list.append(k)

for equ_k in sorted(equ_dict.keys()):
    ext_list.append(equ_k)
print("encode_len:", len(token_list), "decode_len:", len(ext_list))
print("de:", ext_list)

