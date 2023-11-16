from datasets import load_dataset
import random
import json


def parse_data(data_dict):
    instruction = f"""Pick the option that is the best answer to a question corresponding to the following context.
    If the option you pick is 'A', simply reply: 'Answer:A'."""
    opt_to_letter_map = {0:'A', 1:'B', 2:'C', 3:'D'};
    input_txt =  f""" 
    ### context
    {data_dict['context']} 
    
    ### question
    {data_dict['query']}
    
    ###options
    A) {data_dict['options'][0]}
    B) {data_dict['options'][1]}
    C) {data_dict['options'][2]}
    D) {data_dict['options'][3]}""";
    output_opt = opt_to_letter_map[data_dict['correct_option']]
    return {"instruction":instruction,
            "input":input_txt,
            "output":output_opt};

def parse_data_random_perm(data_dict):
    instruction = f"""Pick the option that is the best answer to a question corresponding to the following context.
    If the option you pick is 'A', simply reply: 'Answer:A'."""
    index = [0, 1, 2, 3]
    opt_to_letter_map = {0:'A', 1:'B', 2:'C', 3:'D'};
    random.shuffle(index);
    correct_idx = 0;
    for i in range(0, len(index)):
        opts = index[i];
        if opts == data_dict['correct_option']:
            correct_idx = i
    
    input_txt =  f""" 
    ### context
    {data_dict['context']} 
    
    ### question
    {data_dict['query']}
    
    ###options
    A) {data_dict['options'][index[0]]}
    B) {data_dict['options'][index[1]]}
    C) {data_dict['options'][index[2]]}
    D) {data_dict['options'][index[3]]}""";
    output_opt = opt_to_letter_map[correct_idx];
    return {"instruction":instruction,
            "input":input_txt,
            "output":output_opt};
def get_json_list(dataset, role, random):
    original_data = dataset[role];
    list_data_json = [];
    if (not random):
        for i in range(0, len(original_data)):
            data_dict = original_data[i];
            list_data_json.append(parse_data(data_dict));
    else:
        for i in range(0, len(original_data)):
            data_dict = original_data[i];
            list_data_json.append(parse_data_random_perm(data_dict));
        
    return list_data_json;
    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    dataset = 'lucasmccabe/logiqa';
    data = load_dataset(dataset);
    data_train_val_json = [];
    for i in range(0, 3):
        data_train_json = [];
        data_validation_json = [];

        if(i == 0):
            data_train_json = get_json_list(data, 'train', False);
            data_validation_json = get_json_list(data, 'validation', False);
        else:
            data_train_json = get_json_list(data, 'train', True);
            data_validation_json = get_json_list(data, 'validation', True);
        #data_train_val_json = data_train_json;
        #data_train_val_json.extend(data_validation_json);
        #data_train_val_json_string = json.dumps(data_train_val_json, indent=4)
        #with open('data/logiqa/train_val_' + str(i) + '.json', 'w') as file:
        #    file.write(data_train_val_json_string
        data_train_json.extend(data_validation_json)
        data_train_val_json.extend(data_train_json)
    data_train_val_json_string = json.dumps(data_train_val_json, indent=4)
    with open('evaluate/data/logiqa/train_val.json', 'w') as file:
        file.write(data_train_val_json_string)

    data_test_pre_finetune_eval_json = get_json_list(data, 'test', False)
    data_test_pre_finetune_eval_json = random.sample(data_test_pre_finetune_eval_json, 200)
    data_test_pre_finetune_eval_json_string = json.dumps(data_test_pre_finetune_eval_json, indent=4)
    with open('evaluate/data/logiqa/test_pre_finetune_eval.json', 'w') as file:
        file.write(data_test_pre_finetune_eval_json_string)

    data_test_post_finetune_eval_json = get_json_list(data, 'test', True)
    data_test_post_finetune_eval_json = random.sample(data_test_post_finetune_eval_json, 200)
    data_test_post_finetune_eval_json_string = json.dumps(data_test_post_finetune_eval_json, indent=4)
    with open('evaluate/data/logiqa/test_post_finetune_eval.json', 'w') as file:
        file.write(data_test_post_finetune_eval_json_string)
    
    