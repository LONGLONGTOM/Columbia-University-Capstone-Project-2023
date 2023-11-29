#!pip install jsonargparse
#!pip install -Uqq transformers datasets accelerate bitsandbytes
#!pip install openai
from datasets import load_dataset
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import queue
import threading

# Generating CoT using existing metaphor dataset
def process_data(data_row:dict,
                     client:OpenAI,
                     result_queue:queue)->None:
    try:
        sentence_to_extract = data_row["sentence"]
        new_row = {}
        context = debtor(sentence_to_extract)
        (system_prompt, user_prompt) = generate_prompt(context)
        response_stereotype = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
            )
    
        response_content = response_stereotype.choices[0].message.content
    
        new_row["instruction"] = f"""The input text contains a text called 'literature_context', which contains a sentence. Identify whether the sentence is metaphoric or literal and analyze the metaphor if metaphoric."""
        new_row["input"] = f"""
        literature_context: {context}
        """
        new_row["output"] = response_content

        result_queue.put({"CoT":new_row})
        print("one row done!")
        
    except Exception as e:
        print(f"Error:{e}")
        result_queue.put(None)
# Generating CoT using gpt-3.5 generated dataset
def generate_metaphor(client, result_queue):
    try:
        new_row = {}
        (system_prompt, user_prompt) = generate_prompt_AI()
        response_stereotype = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
            )
    
        response_content = response_stereotype.choices[0].message.content
        sentences = response_stereotype.choices[0].message.content.split('.')

        # Extract the first sentence (assuming the paragraph contains at least one sentence)
        first_sentence = sentences[0].strip()
        new_row["instruction"] = f"""The input text contains a text called 'literature_context', which contains a sentence. Identify whether the sentence is metaphoric or literal and analyze the metaphor if metaphoric."""
        new_row["input"] = f"""
        literature_context: {first_sentence}
        """
        new_row["output"] = response_content

        result_queue.put({"CoT":new_row})
        print("one row done!")
    except Exception as e:
        print(f"Error:{e}")
        result_queue.put(None)
        
def generate_prompt_AI():
    system_prompt = f"""You are a literature expert in creative writing and is good at distinguish and explaining metaphor in context, which sentence is metaphoric, and what is expressed by the metaphor, or why is the content not metaphoric."""
    user_prompt = f"""Please randomly write a literal or metaphoric sentence in 50/50 chance, 
    then generate an Chain of Thought explanation of whether the sentence is metaphoric. If metaphoric, please analyze the metaphor.
    Your response should begin with repeating the context and explaining whether the content is metaphoric or not.
    Make sure your language is simple and brief."""
    return (system_prompt, user_prompt)

def generate_prompt(context):

    system_prompt = f"""You are a literature expert in creative writing and is good at distinguish and explaining metaphor in context, which sentence is metaphoric, and what is expressed by the metaphor, or why is the content not metaphoric."""
    user_prompt = f"""The following content might or might not be metaphoric: {context}
    Generate an Chain of Thought explanation of whether the content is metaphoric. If metaphoric, please analyze the metaphor.
    Your response should begin with repeating the context and explaining whether the content is metaphoric or not.
    Make sure your language is simple and brief."""
    return (system_prompt, user_prompt)
 
def generate_CoT_From_GPT(
    api_key:str = "",
    destination_path:Path = Path("metaphor_detection/data"),
    out_file_name:str = "Metaphor_CoT_explanation",
) -> None:
    dataset = load_dataset("CreativeLang/moh_metaphor")['train'] # 
    client = OpenAI(api_key = api_key)
    
    result_queue = queue.Queue()
    threads = []
    for i in range(1000):
        thread = threading.Thread(target = generate_metaphor, args = (client, result_queue))
        threads.append(thread)
    for i in tqdm(range(0, len(dataset)), desc = "Number of samples evaluated:"):
        data_row = dataset[i]
        thread = threading.Thread(target = process_data, args = (data_row, client, result_queue))
        threads.append(thread)
    
    print("Done loading threads")
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
    with open(out_file_path_json_list_total_string, 'w+') as file:
        file.write(json_list_total_string)
      
    return

def debtor(str):
  return (str.replace("<b>", "")).replace("<b>","")
print("like that?")
if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    CLI(generate_CoT_From_GPT)