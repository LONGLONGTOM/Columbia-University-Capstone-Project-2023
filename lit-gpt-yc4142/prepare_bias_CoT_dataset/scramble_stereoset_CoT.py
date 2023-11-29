import json

from pathlib import Path
import random


def scramble(
    source_path:Path = Path("prepare_bias_CoT_dataset/data"),
    data_file_name:str = "bias_CoT_explanation_multi_thread_total.json",
    destination_path:Path = Path("prepare_bias_CoT_dataset/data"),
    out_file_name:str = "bias_CoT_explanation_multi_thread_scrambled.json",
) ->None:
    original_data_path = source_path / data_file_name
    with open(original_data_path, mode = "r") as file:
        original_data = json.load(file)
    out_json_list = []
    for data_pairs in original_data:
        out_json_list.append(data_pairs["stereotype_CoT"])
        out_json_list.append(data_pairs["anti_stereotype_CoT:"])
    random.shuffle(out_json_list)
    destination_file = destination_path / out_file_name
    json_object = json.dumps(out_json_list, indent=4)
    with open(destination_file, 'w') as file:
        file.write(json_object)
    return 


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(scramble)