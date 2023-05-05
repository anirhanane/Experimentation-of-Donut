#import librairies
from shutil import copyfile,move
from git import Repo
import json
from pathlib import Path
import shutil
import json
from pathlib import Path
from datasets import load_dataset,load_from_disk
import random
from transformers import DonutProcessor
import re
from typing import Any, Dict, List, Tuple, Union, Optional
import os

new_special_tokens = [] # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>" # eos token of tokenizer

#def connexion_HF():
    # Use the Hugging Face Hub as a remote model versioning service
    #from huggingface_hub import notebook_login
    #notebook_login()

#def connexion_wandb():
    #wandb.login()


def clone_dataset_sroie(repo_dir):
    """
    Clone SROIE dataset
    """
    # clone repository
    Repo.clone_from("https://github.com/zzzDavid/ICDAR-2019-SROIE.git",repo_dir)
    # copy data
    move("./dataset/ICDAR-2019-SROIE/data", "./dataset/sroie")


def create_metadata():
    """
    Create a metadata.json file that contains the information about the images including the Gth OCR-text.
    This is necessary for the imagefolder feature of datasets
    """
    # define paths
    base_path = Path("data")
    metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")
    # define metadata list
    metadata_list = []

    # parse metadata
    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            # load json file
            data = json.load(json_file)
            # create "text" column with json string
            text = json.dumps(data)
            # add to metadata list if image exists
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                metadata_list.append({"text":text,"file_name":f"{file_name.stem}.jpg"})
            
    # write jsonline file
    with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    # remove old meta data
    shutil.rmtree(metadata_path)


def load_database_sroie():
    """
    Load the dataset using the imagefolder feature of datasets
    """
    # define paths
    base_path = Path("dataset/sroie")
    metadata_path = base_path.joinpath("key")
    image_path = base_path.joinpath("img")

    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

    print(f"Dataset has {len(dataset)} images")
    print(f"Dataset features are: {dataset.features.keys()}")
    return dataset


def view_dataset(dataset):
    """
    Look at the dataset
    """
    random_sample = random.randint(0, len(dataset))

    print(f"Random sample is {random_sample}")
    print(f"OCR text is {dataset[random_sample]['text']}")
    dataset[random_sample]['image'].resize((250,400))
    #OCR text is {"company": "LIM SENG THO HARDWARE TRADING", "date": "29/12/2017", "address": "NO 7, SIMPANG OFF BATU VILLAGE, JALAN IPOH BATU 5, 51200 KUALA LUMPUR MALAYSIA", "total": "6.00"}


def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj
    
def preprocess_documents_for_donut(sample):
    """
    Prepare Datset for Donut
    """
    # create Donut-style input
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGB
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}

def preprocess_documents_for_donut(dataset):
    """
    Process Datset for Donut
    """
    proc_dataset = dataset.map(preprocess_documents_for_donut)

    print(f"Sample: {proc_dataset[45]['text']}")
    print(f"New special tokens: {new_special_tokens + [task_start_token] + [eos_token]}")
    return proc_dataset

def load_DonutProcessor():
    """Load DonutProcessor, add new special tokens and adjust the size of the images 
    when processing from [1920, 2560] to [720, 960] to need less memory and have faster training
    """
    # Load processor
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    # add new special tokens to tokenizer
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})

    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # resizing the image to smaller sizes from [1920, 2560] to [960,1280]
    processor.feature_extractor.size = [720,960] # should be (width, height)
    processor.feature_extractor.do_align_long_axis = False
    return processor

def transform_and_tokenize(sample, processor, split="train", max_length=512, ignore_id=-100):
    """Transform image to tensor and Tokenize the ground truth
    """
    # create tensor from image
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}

def processed_dataset(proc_dataset) :
    # need at least 32-64GB of RAM to run this
    processed_dataset = proc_dataset.map(transform_and_tokenize,remove_columns=["image","text"])
    return processed_dataset

def save_Dataset_Processor(processed_dataset,processor):
    """
    Save processed data and processor in case of error later
    """
    processed_dataset.save_to_disk("processed_dataset")
    processor.save_pretrained("processor")

def load_Dataset_Processor():
    """
    Load the processed dataset from disk in case of error later
    """
    processed_dataset = load_from_disk("processed_dataset")
    processor = DonutProcessor.from_pretrained("processor")
    return processed_dataset,processor

def split_processed_dataset(dataset):
    """
    Split the dataset into train and validation sets
    """

    processed_dataset = processed_dataset.train_test_split(test_size=0.1)
    print(processed_dataset)
    return processed_dataset

def token2json(model,processor, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in processor.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)


def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)
    
