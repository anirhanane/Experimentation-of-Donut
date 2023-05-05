import torch
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, DonutProcessor, logging
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer,Trainer,TrainingArguments
import re
from PIL import Image
import torch
import random
import numpy as np
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
import zss
from datasets import load_dataset
from nltk import edit_distance
from torch.utils.data import Dataset
from zss import Node
from transformers.file_utils import ModelOutput
from .utils import token2json,save_json
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def finetunning(processed_dataset,processor):
    """
    Finetuning with SROIE dataset
    """
    # Load model from huggingface.co
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Resize embedding layer to match vocabulary size
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")
    # Adjust our image size and output sequence lengths
    model.config.encoder.image_size = processor.feature_extractor.size[::-1] # (height, width)
    model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))

    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

    # move the model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    #define the hyperparameters (Seq2SeqTrainingArguments) we want to use for our training.

    # hyperparameters used for multiple args
    hf_repository_id = "donut-base-sroie"
    # Arguments for training Tunning
    training_args = Seq2SeqTrainingArguments(
        output_dir=hf_repository_id,
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=15,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        #save_total_limit=2,
        evaluation_strategy="no",
        save_strategy="steps",
        #save_steps=1,
        predict_with_generate=True,
        # push to hub parameters
        report_to="wandb",
        push_to_hub=True,
        #log_on_each_node=True,
        #disable_tqdm =False,
        hub_strategy="every_save",
        hub_model_id=hf_repository_id,
        hub_token=HfFolder.get_token(),
    )
        # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
    )
    # Start training Tunning
    trainer.train()
    # Save processor and create model card into Hugging Face
    processor.save_pretrained(hf_repository_id)
    trainer.create_model_card()
    trainer.push_to_hub()

def run_prediction(sample, model, processor):
    """
    Run prediction for a specific sample with the pretrained model and processor
    """
    # prepare inputs
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    # load reference target
    target = processor.token2json(sample["target_sequence"])
    return prediction, target

def inference(
        model,processor,
        image_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image_tensors: (1, num_channels, height, width)
                fed the image_tensor
            prompt_tensors: (1, sequence_length)
                fed the prompt_tensor
        """


        if model.device.type == "cuda":  # half is not compatible in cpu implementation.
            #image_tensors = image_tensors.half() #equivalent to self.to(torch.float16)
            image_tensors = image_tensors.to(device)
        prompt_tensors = processor.tokenizer("<s>", add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(device)

        # get decoder output
        decoder_output = model.generate(
            image_tensors,
            decoder_input_ids=prompt_tensors,
            max_length=model.decoder.config.max_position_embeddings,#model.config.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {"predictions": list()}
        for seq in processor.batch_decode(decoder_output.sequences):
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(token2json(model,processor,seq))
            else:
                output["predictions"].append(seq)

        # process output
        #prediction = processor.batch_decode(decoder_output.sequences)[0]
        #prediction = processor.token2json(prediction)
        #print(prediction)
        #output["predictions"]=prediction

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

def sample_inference(test_sample, path_pretrained_model):
    """
    Test the pretrained model "Hanoun/donut-base-sroie" based on a random document image from the test set "processed_dataset["test"]"

    """
    # hidde logs
    logging.disable_default_handler()


    # Load our model from Hugging Face
    processor = DonutProcessor.from_pretrained(path_pretrained_model)
    model = VisionEncoderDecoderModel.from_pretrained(path_pretrained_model)

    # Move model to GPU
    model.to(device)

    # Run Prediction
    prediction, target = run_prediction(test_sample,model, processor)
    print(f"Reference:\n {target}")
    print(f"Prediction with run prediction function:\n {prediction}")
    
    # Run inference
    image_tensors=torch.tensor(test_sample["pixel_values"]).unsqueeze(0)
    prompt_tensors=None
    predic_json = inference(model,processor, image_tensors,True,False)
    print(f"Prediction with inference function:\n {predic_json}")

def accuracy(model,processor,dataset_test):
  """
  Evalute the model on the test set
  """
  # define counter for samples
  true_counter = 0
  total_counter = 0

  # iterate over dataset
  for sample in tqdm(dataset_test):
    prediction, target = run_prediction(sample,model, processor)
    for s in zip(prediction.values(), target.values()):
      if s[0] == s[1]:
        true_counter += 1
      total_counter += 1
  accuracy=(true_counter/total_counter)*100

  print(f"Accuracy: {accuracy}%")
  return accuracy


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )

def Load_error_per_epoch_step(dataset_train,directory):
    """
    Based on checkpoints calculate the residual error per epoch or per step

    """  

    weights=[]
    errors=[]

    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.startswith('check'):
                checkpoint_dir=os.path.join(root, dir)
                processor = DonutProcessor.from_pretrained(directory)
                model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                error=100-accuracy(model,processor,dataset_train)
                errors.append(error)
    return errors

def Load_weight_per_epoch_step(layer,dataset_train,directory,i,j):
    """
    Based on checkpoints  per epoch or  per step extract weights of the model in predefined layer

    """  

    weights=[]

    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.startswith('check'):
                checkpoint_dir=os.path.join(root, dir)
                processor = DonutProcessor.from_pretrained(directory)
                model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                if os.path.exists(os.path.join(checkpoint_dir,"pytorch_model.bin")):
                    try:
                        model_state_dict=torch.load(os.path.join(checkpoint_dir,"pytorch_model.bin"))
                    except EOFError:
                        print("corrupted file")
                        weights.append(None)
                        pass
                    weights.append((model_state_dict[layer][i][j]))
                else:
                    weights.append(None)
    weights_Fin = [i.item() if i is not None else None for i in weights]
    return weights_Fin

def analyze_PretrainedModel_Param(dataset_train,directory,layer,i,j):
    """
    Extract weights of the model in each layer(i=[0-input-size] ,j=[0-hiddensize]) and calculate the residual error per epoch 
    And visualize de correlation between weights and residual error
    ex: directory='donut-base-sroie_TunningParam'
    """

    directory = 'donut-base-sroie_TunningParam'

    weights_Fin=Load_weight_per_epoch_step(layer,dataset_train,directory, i,j)
    errors=Load_error_per_epoch_step(dataset_train,directory)
    
    plt.plot(weights_Fin,errors)




def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def test(path,dataset,task_name,save_path):
  processor = DonutProcessor.from_pretrained(path)
  pretrained_model = VisionEncoderDecoderModel.from_pretrained(path)
  if torch.cuda.is_available():
    #pretrained_model.half()
    pretrained_model.to("cuda")

  pretrained_model.eval()
  predictions = []
  ground_truths = []
  accs = []

  evaluator = JSONParseEvaluator()

  for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
      #print(sample)
      ground_truth = sample["target_sequence"]#json.loads(

      if task_name == "docvqa":
          task_prompt=f"<s_{task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>"
          prompt = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
          output = inference(pretrained_model,processor,
              torch.tensor(sample["pixel_values"]).unsqueeze(0),prompt
            )["predictions"][0]
      else:
          task_prompt = "<s>"
          prompt = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
          output = inference(pretrained_model,processor,torch.tensor(sample["pixel_values"]).unsqueeze(0), prompt)["predictions"][0]
          

      if task_name == "rvlcdip":
          gt = ground_truth["gt_parse"]
          score = float(output["class"] == gt["class"])
      elif task_name == "docvqa":
          # Note: we evaluated the model on the official website.
          # In this script, an exact-match based score will be returned instead
          gt = ground_truth["gt_parses"]
          answers = set([qa_parse["answer"] for qa_parse in gt])
          score = float(output["answer"] in answers)
      else:     
          gt = ground_truth
          #print(gt)
          #print(output)
          score = evaluator.cal_acc(output, gt)

      accs.append(score)

      predictions.append(output)
      ground_truths.append(gt)

  scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
    }
  print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

  if save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        save_json(save_path,scores)

  return predictions