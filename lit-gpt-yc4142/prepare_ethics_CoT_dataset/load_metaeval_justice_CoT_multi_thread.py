from datasets import load_dataset
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import queue
import threading
import csv
import random 


def process_metaeval_justice_data(justice_row:dict,
                     client:OpenAI,
                     result_queue:queue)->None:
    try:
        acceptable = justice_row["label"]  
        scenario = justice_row["scenario"]
 
        new_row_justice = {}
        system_prompt_stereotype = ""
        user_prompt_stereotype = ""
        if (acceptable == '0'):
            (system_prompt_stereotype, user_prompt_stereotype) = generate_prompt_unacceptable(scenario)
        else:
            (system_prompt_stereotype, user_prompt_stereotype) = generate_prompt_acceptable(scenario)

        response_justice = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_stereotype},
                {"role": "user", "content": user_prompt_stereotype},
                ]
            )
    
        response_justice_content = response_justice.choices[0].message.content
    
        new_row_justice["instruction"] = f"""Determine if the following scenario is acceptable according to justice or not."""
        new_row_justice["input"] = f"""
        scenario: {scenario}
        """
        new_row_justice["output"] = response_justice_content
        
      

       
        result_queue.put(new_row_justice)
        
    except Exception as e:
        print(f"Error:{e}")
        result_queue.put(None)

def generate_prompt_acceptable(
    scenario:str
)->(str, str):

    system_prompt = f"""You are a language model that is well equipped with knowledge of social justice. You know what constituses justice is impartiality, that similar agents should be treated the same way, and desert, that agents should get what they deserve."""
    user_prompt = f"""The following content contains an ethical scenario related to justice: {scenario}. 
    The scenario is deemed ethically acceptable.
    Generate an Chain of Thought explanation on why the scenario is ethically acceptable by justice, such as how the scenario demonstrates impartiality and/or desert and therefore demonstrates justice. If some elements of justice is not involved in the scenario, you don't need to mention them.
    Start your response with "The scenario is ethically acceptable." and Chain of Thought explanation following it without explicitly mentioning you are applying Chain of Thought."""
    return (system_prompt, user_prompt)
    
def generate_prompt_unacceptable(
    scenario:str,
)->(str, str):
    system_prompt = f"""You are a language model that is well equipped with knowledge of social justice. You know what constituses justice is impartiality, that similar agents should be treated the same way, and desert, that agents should get what they deserve."""
    user_prompt = f"""The following content contains an ethical scenario related to justice: {scenario}. 
    The scenario is deemed ethically unacceptable.
    Generate an Chain of Thought explanation on why the scenario is ethically unacceptable by justice, such as how the scenario fail to demonstrates impartiality and/or desert and therefore does not demonstrate justice. If some elements of justice is not involved in the scenario, you don't need to mention them.
    Start your response with "The scenario is ethically unacceptable." and Chain of Thought explanation following it without explicitly mentioing you are applying Chain of Thought."""
    return (system_prompt, user_prompt)

def generate_CoT_From_GPT(
    api_key:str = "",
    data_file_path:Path = Path("/prepare_ethics_CoT_dataset/ethics_raw_data/justice/justice_train.csv"),
    destination_path:Path = Path("prepare_ethics_CoT_dataset/data"),
    out_file_name:str = "ethics_justice_CoT_explanation.json",
    sample_number:int = 2000,
    acceptable_unacceptable_ratio:float = 0.5,
) -> None:
    raw_dataset = None
    with open(data_file_path, mode = "r", encoding = 'utf-8') as file:
        csv_reader = csv.DictReader(file)
        raw_dataset = list(csv_reader)
    
    assert sample_number <= len(raw_dataset), "number samples to pick from raw dataset should be lower or equal to the size of entire raw dataset."
    data_acceptable = []
    data_unacceptable = []
    for data_rows in raw_dataset:
        if (data_rows["label"] == '0'):
            data_acceptable.append(data_rows)
        else:
            data_unacceptable.append(data_rows)
    acceptable_size = int(sample_number * acceptable_unacceptable_ratio)
    unacceptable_size = int(sample_number - acceptable_size)

    index_to_pick_acceptable = random.sample(range(len(data_acceptable)), acceptable_size)
    index_to_pick_unacceptable = random.sample(range(len(data_unacceptable)), unacceptable_size)
    dataset = []
    for indexes in index_to_pick_acceptable:
        dataset.append(data_acceptable[indexes])
    for indexes in index_to_pick_unacceptable:
        dataset.append(data_unacceptable[indexes])
    random.shuffle(dataset)
    client = OpenAI(api_key = api_key)
    
    result_queue = queue.Queue()
    threads = []

    for i in tqdm(range(0, len(dataset)), desc = "Number of samples evaluated:"):
        justice_row = dataset[i]
        thread = threading.Thread(target = process_metaeval_justice_data, args = (justice_row, client, result_queue))
        threads.append(thread)

    for thread in threads:
        thread.start()
 
    for thread in threads:
        thread.join()

    json_list_total = []
    while not result_queue.empty():
        CoT_reponse = result_queue.get()
        if CoT_reponse is not None:
            json_list_total.append(CoT_reponse)


    json_list_total_string = json.dumps(json_list_total, indent=4)
    out_file_path_json_list_total_string = destination_path / (out_file_name)
    with open(out_file_path_json_list_total_string, 'w') as file:
        file.write(json_list_total_string)
      
    return

    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(generate_CoT_From_GPT)