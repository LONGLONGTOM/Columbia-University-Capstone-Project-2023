import json

from pathlib import Path
import random


def scramble(
    source_path:Path = Path("prepare_ethics_CoT_dataset/data"),
    commonsense_data_file_name:str = "ethics_commonsense_CoT_explanation.json",
    deontology_data_file_name:str = "ethics_deontology_CoT_explanation.json",
    justice_data_file_name:str = "ethics_justice_CoT_explanation.json",
    destination_path:Path = Path("prepare_ethics_CoT_dataset/data"),
    out_file_name:str = "ethics_CoT_explanation_scrambled.json",
) ->None:
    commonsense_data_path = source_path / commonsense_data_file_name
    with open(commonsense_data_path, mode = "r") as file:
        commonsense_data = json.load(file)

    deontology_data_path = source_path / deontology_data_file_name
    with open(deontology_data_path, mode = "r") as file:
        deontology_data = json.load(file)

    justice_data_path = source_path / justice_data_file_name
    with open(justice_data_path, mode = "r") as file:
        justice_data = json.load(file)
    out_json_list = commonsense_data
    out_json_list.extend(deontology_data)
    out_json_list.extend(justice_data)
     
    random.shuffle(out_json_list)
    destination_file = destination_path / out_file_name
    json_object = json.dumps(out_json_list, indent=4)
    with open(destination_file, 'w') as file:
        file.write(json_object)
    return 


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(scramble)