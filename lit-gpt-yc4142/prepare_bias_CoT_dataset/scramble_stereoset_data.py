import json

from pathlib import Path
import random


def scramble(
    source_path:Path = Path("prepare_bias_CoT_dataset/data"),
    CoT_data_file_name:str = "bias_CoT_reasoning.json",
    non_CoT_data_file_name:str = "bias_non_CoT_reasoning.json",
    destination_path:Path = Path("prepare_bias_CoT_dataset/data"),
    CoT_out_file_name_train:str = "bias_CoT_reasoning_scrambled_train.json",
    CoT_out_file_name_test:str = "bias_CoT_reasoning_scrambled_test.json",
    non_CoT_out_file_name_train:str = "bias_non_CoT_reasoning_scrambled_train.json",
    non_CoT_out_file_name_test:str = "bias_non_CoT_reasoning_scrambled_test.json",
    test_ratio:float = 0.03865
) ->None:
    assert(test_ratio < 1 and test_ratio > 0), "test sample ratio should be larger than 0 and smaller than 1."
   
    CoT_data_path = source_path / CoT_data_file_name
    with open(CoT_data_path, mode = "r") as file:
        CoT_data = json.load(file)
    CoT_json_list = []
    for data_triplets in CoT_data:
        CoT_json_list.append(data_triplets["stereotype_CoT"])
        CoT_json_list.append(data_triplets["anti_stereotype_CoT"])
        CoT_json_list.append(data_triplets["no_stereotype_CoT"])
    
    non_CoT_data_path = source_path / non_CoT_data_file_name
    with open(non_CoT_data_path, mode = "r") as file:
        non_CoT_data = json.load(file)
    non_CoT_json_list = []
    for data_triplets in non_CoT_data:
        non_CoT_json_list.append(data_triplets["stereotype_CoT"])
        non_CoT_json_list.append(data_triplets["anti_stereotype_CoT"])
        non_CoT_json_list.append(data_triplets["no_stereotype_CoT"])
    
    shuffled_index = [i for i in range(0, len(CoT_json_list))];
    random.shuffle(shuffled_index)

    train_size = len(CoT_json_list) - int(len(CoT_json_list) * test_ratio)
    CoT_json_list_out_train = []
    non_CoT_json_list_out_train = []
    CoT_json_list_out_test = []
    non_CoT_json_list_out_test = []
    for i in range(0, len(CoT_json_list)):
        if (i < train_size):
            CoT_json_list_out_train.append(CoT_json_list[shuffled_index[i]])
            non_CoT_json_list_out_train.append(non_CoT_json_list[shuffled_index[i]])
        else:
            CoT_json_list_out_test.append(CoT_json_list[shuffled_index[i]])
            non_CoT_json_list_out_test.append(non_CoT_json_list[shuffled_index[i]])

    destination_file_CoT_train = destination_path / CoT_out_file_name_train
    json_object_CoT_train = json.dumps(CoT_json_list_out_train, indent=4)
    with open(destination_file_CoT_train, 'w') as file:
        file.write(json_object_CoT_train)

    destination_file_non_CoT_train = destination_path / non_CoT_out_file_name_train
    json_object_non_CoT_train = json.dumps(non_CoT_json_list_out_train, indent=4)
    with open(destination_file_non_CoT_train, 'w') as file:
        file.write(json_object_non_CoT_train)

    destination_file_CoT_test = destination_path / CoT_out_file_name_test
    json_object_CoT_test = json.dumps(CoT_json_list_out_test, indent=4)
    with open(destination_file_CoT_test, 'w') as file:
        file.write(json_object_CoT_test)

    destination_file_non_CoT_test = destination_path / non_CoT_out_file_name_test
    json_object_non_CoT_test = json.dumps(non_CoT_json_list_out_test, indent=4)
    with open(destination_file_non_CoT_test, 'w') as file:
        file.write(json_object_non_CoT_test)
    
    


    return 


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(scramble)