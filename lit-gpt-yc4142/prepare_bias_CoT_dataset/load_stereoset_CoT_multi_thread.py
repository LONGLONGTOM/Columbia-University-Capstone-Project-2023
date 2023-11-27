from datasets import load_dataset
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import queue
import threading


def process_stereoset_data(stereodata_row:dict,
                     client:OpenAI,
                     result_queue:queue)->None:
    try:
        bias_type = stereodata_row["bias_type"]
        
        sentence_to_extract = stereodata_row["sentences"]

        stereotype_index = 0
        anti_stereotype_index = 0
        for j in range(0, len(sentence_to_extract["gold_label"])):
            label = sentence_to_extract["gold_label"][j];
            if(label == 0):
                anti_stereotype_index = j
            if(label == 1):
                stereotype_index = j
        stereotype_sentence = sentence_to_extract["sentence"][stereotype_index]
        anti_stereotype_sentence = sentence_to_extract["sentence"][anti_stereotype_index]
 
        new_row_stereotype = {}
        stereotype_context = stereodata_row["context"] + " " + stereotype_sentence 
        (system_prompt_stereotype, user_prompt_stereotype) = generate_prompt_stereotype(bias_type, stereotype_context)
        response_stereotype = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_stereotype},
                {"role": "user", "content": user_prompt_stereotype},
                ]
            )
    
        response_stereotype_content = response_stereotype.choices[0].message.content
    
        new_row_stereotype["instruction"] = f"""The input text contains a text called 'stereotype_context', which contains a {bias_type} stereotype against {stereodata_row['target']}. Explain what type of stereotype is involved and why such stereotype against {stereodata_row['target']} is harmful."""
        new_row_stereotype["input"] = f"""
        stereotype_context: {stereotype_context}
        """
        new_row_stereotype["output"] = response_stereotype_content
        
      


        new_row_anti_stereotype = {}
        anti_stereotype_context = stereodata_row["context"] + " " + anti_stereotype_sentence
        (system_prompt_anti_stereotype, user_prompt_anti_stereotype) = generate_prompt_anti_stereotype(bias_type, stereotype_context, anti_stereotype_context)
        response_anti_stereotype = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_anti_stereotype},
                {"role": "user", "content": user_prompt_anti_stereotype},
                ]
            )
        
        response_anti_stereotype_content = response_anti_stereotype.choices[0].message.content
        
        new_row_anti_stereotype["instruction"] = f"""The input text contains a text called 'stereotype_context', which contains a {bias_type} stereotype against {stereodata_row['target']}, and a text called 'anti_stereotype_context', which contains a statement against the {bias_type} stereotype in 'stereotype_context'. Explain how the 'anti_stereotype_context' counters {bias_type} stereotype against {stereodata_row['target']} in 'stereotype_context'."""
        new_row_anti_stereotype["input"] = f"""
        stereotype_context: {stereotype_context}
        anti_stereotype_context: {anti_stereotype_context}
        """
        new_row_anti_stereotype["output"] = response_anti_stereotype_content

        #result_queue.put(new_row_stereotype)
        #result_queue.put(new_row_anti_stereotype)
        result_queue.put({"bias_type":bias_type,
                          "stereotype_CoT":new_row_stereotype,
                          "anti_stereotype_CoT:":new_row_anti_stereotype})
        
    except Exception as e:
        print(f"Error:{e}")
        result_queue.put(None)

def generate_prompt_stereotype(
    bias_type:str, 
    context_stereotype:str
)->(str, str):

    system_prompt = f"""You are an expert in {bias_type} bias and is good at explaining bias in context, what harm {bias_type} bias can do, and what to do to avoid {bias_type} bias."""
    user_prompt = f"""The following content contains {bias_type} stereotype related content: {context_stereotype}
    Generate an Chain of Thought explanation of the {bias_type} bias involved in the context.
    Your response should begin with repeating the context and explaining what type of {bias_type} bias is involved in the context.
    Make sure your language is simple and brief."""
    return (system_prompt, user_prompt)
    
def generate_prompt_anti_stereotype(
    bias_type:str, 
    context_stereotype:str, 
    context_anti_stereotype:str
)->(str, str):
    system_prompt = f"""You are an expert in {bias_type} bias and is good at explaining bias in context, what harm {bias_type} bias can do and what to do to avoid {bias_type} bias."""
    user_prompt = f"""The following is a stereotype context containing {bias_type} bias: {context_stereotype}
    The following is an anti-stereotype context against the {bias_type} bias mentioned: {context_anti_stereotype}
    Generate an Chain of Thought explanation of how the context is against {bias_type} bias mentioned.
    Your response should begin with repeating the stereotype context, stating how the anti-stereotype context is
    against the stereotype context, and what stereotype is involved in both context.
    Make sure your language is simple and brief."""
    return (system_prompt, user_prompt)

def generate_CoT_From_GPT(
    api_key:str = "",
    destination_path:Path = Path("data/stereoset_CoT"),
    out_file_name:str = "bias_CoT_explanation",
) -> None:
    dataset = load_dataset("stereoset", "intersentence")['validation']
    client = OpenAI(api_key = api_key)
    
    result_queue = queue.Queue()
    threads = []

    for i in tqdm(range(0, len(dataset)), desc = "Number of samples evaluated:"):
        stereoset_row = dataset[i]
        thread = threading.Thread(target = process_stereoset_data, args = (stereoset_row, client, result_queue))
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
    out_file_path_json_list_total_string = destination_path / (out_file_name + "_total.json")
    with open(out_file_path_json_list_total_string, 'w') as file:
        file.write(json_list_total_string)
      
    return

    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(generate_CoT_From_GPT)
