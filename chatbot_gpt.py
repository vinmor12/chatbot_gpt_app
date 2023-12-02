# CHAT BOT with GPT-2 & GPT-NEO
# Vincenzo - y.2023-v.0.1
# ----------------------------------------------------
# Developed with Python 3.7
# Tested on CPU with OpenVINO 2022.3 for Windows
# 
# PREREQUISITES
# install Gradio for Interactive Inference with:
# pip install gradio
# ----------------------------------------------------

# STARTING
import os
os.system('cls')
print("\n"," "*(20-round(22/2)),"WELCOME to CHATBOT APP")
print("-"*40)
print(" "*(20-round(12/2)), "Starting ...\n")
from colorama import Fore, Back, Style
print(Fore.GREEN +"Developed with Python 3.10\nTested on CPU with OpenVINO 2022.3\nfor Windows\n\nvinmor12 - y.2023-v.0.1")
print(Fore.WHITE +"-"*40)
from gradio import Blocks, Chatbot, Textbox, Row, Column
import ipywidgets as widgets
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
from openvino.runtime import serialize
from openvino.tools import mo
from transformers.onnx import export, FeaturesManager
from openvino.runtime import Core
from typing import List, Tuple
import numpy as np
import time
import sys
style = {'description_width': 'initial'}
model_name = widgets.Select(
            options=['GPT-2', 'GPT-Neo'],
            value='GPT-2',
            description='Select Model:',
            disabled=False
        )
widgets.VBox([model_name])
os.system('cls')
# to resize window console uncomment following line:
# os.system('mode con: cols=40 lines=80')

# DRAW SPRITE FUNCTION
def draw_sprite(sprite):
    #sprite = 1 #2
    if sprite == 1:
        print(Fore.GREEN + "\n"," "*(20-round(11/2)),"CHATBOT APP")
        print(Fore.WHITE + "-"*40)
        print(Fore.GREEN + "           ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("         ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("       ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("    ▒▒▒▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒▒▒▒▒▒▒ ")
        print("    ▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▒▒ ")
        print("    ▒▒▓███████▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓█████▓▒▒ ")
        print("    ▒▒▓█████████▓▓▒▒▒▒▒▒▒▓▓██████▓▓▒▒ ")
        print("    ▒▒▒▓██████████▓▒▒▒▒▓████████▓▓▒▒ ")
        print("    ▒▒▒▒▓▓▓████▓▓▓▓▒▒▒▒▒▓▓███▓▓▓▒▒▒▒ ")
        print("     ▒▒▒▒▒▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("       ▒▒▒▒▒▒▒▒▒▒▓█▒▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("        ▒▓▒▒▒▒▒▒▓▓█▒▒▒▒▒▒▒▒▒▒▒▒▒ ")
        print("          ▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░ ")    
        print("           ▓▒▒▒▒▓█████▓▒▒▒▒▒▒░ ")
        print("             ▒▒▓▒▒▒▒▒▒▒▓▒▒░░ ")     
        print("                ▒▒▒▒▒▒▒ ")
        print(Fore.WHITE + "-"*40)
    else:
        print(Fore.GREEN + "\n"," "*(20-round(11/2)),"CHATBOT APP")
        print(Fore.WHITE + "-"*40)
        print(Fore.GREEN +"          ███████▓▒░░░░░░░▒▒▒█▒ ")
        print("       ▓████▓▒▒▒▒░░░░░░░░░░░░▒░▒ ")
        print("     ▒████▓▒░░░░░░░░░░░░░░░░░░░▓██▒ ")
        print("    ▓████▒░░░░░░░░░░░░░░░░░░░░░▓▓██▓ ")
        print("   ▒███▓░░░▒▒▒▒▒░░░░░░░░▒▒▒▒▒▒░▒███▓ ")
        print("   ▒███▓░░░▒▒▒▒▒░░░░░░░░▒▒▒▒▒▒░▒███▓ ")
        print("   ▒███▓▓████▓▓████▓▒░▒██████████░▓██▒ ")
        print("   ▒███▓▒▒▓▓▓█▓▓▓▒░░░░░▒▓▓▓▓▓▒░░░░███▒ ")
        print("   ▓███▓░▒▒░▒▒░░▒▒░░░░░░░░▒▒▒░░░░░▓██ ")
        print("   ▒███▒░░░░░░░░░░░▒░░░░░░░░░░░░░░██▒ ")
        print("     ██▒░░░░░░░░░▓▓▒░░▓▒░░░░░░░░░░█ ")
        print("     ▒█▓░░░░░░░░▒░░░▒░░░░▒░░░░░░░▒ ")
        print("     ▒█▓░░░░░░░▒░░░░▒░░░░░▒░░░░░░▒ ")
        print("        █▒░░░░▒▓██▓▓▒▒▒▒▒▒░░░░░░ ")
        print("         ▓▓▒░░░▒░░░░░░░░░░▒░░░ ")
        print("           ▓▓░░░░░░░░░░░░░░░░ ")
        print("             ▒▒▒▓▓▓▓▒▒▒░░░░ ")
        print(Fore.WHITE + "-"*40)

# MAIN MENU FUNCTION
# main menu
def menu_fun():
    os.system('cls')
    draw_sprite(1)
    print(" "*(20-round(9/2)),"MAIN MENU")
    print("-"*40)
    print("Select the Options:\n 1 - Get Pytorch Models\n 2 - Covert Models in IR\n 3 - Run Inference\nCTRL+C to Quit")
    print("-"*40)
    flag = int(input(">> "))
    if flag == 1:
        get_models_fun()
    elif flag == 2:
        convert_models_fun()
    elif flag == 3:
        inference_fun()

# DOWNLOAD MODELS FUNCTION
# download gpt-2 and gpt-neo pytorch models 
# and tokenizer from huggingface
def get_models_fun():
    os.system('cls')
    draw_sprite(2)
    print(" "*(20-round(18/2)),"GET PYTORCH MODELS")
    print("-"*40)
    print("Select the Model:\n 1 - GPT-Neo\n 2 - GPT-2\n 3 - Back")
    print("-"*40)
    flag = int(input(">> "))
    enable = 1
    if flag == 2:
        model_name.value = 'GPT-2'
    elif flag == 1:
        model_name.value = 'GPT-Neo'
    else:
        menu_fun()
        enable = 0
    if enable == 1:
        os.system('cls')
        draw_sprite(1)
        print("Download ...\n")
        if model_name.value == 'GPT-2':
            pt_model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif model_name.value == 'GPT-Neo':
            pt_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
            tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')

# CONVERT MODELS FUNCTION
# convert pytorch model to OpenVINO IR
def convert_models_fun():
    os.system('cls')
    draw_sprite(2)
    print(" "*(20-round(20/2)),"CONVERT MODELS IN IR")
    print("-"*40)
    print("Select the Model:\n 1 - GPT-Neo\n 2 - GPT-2\n 3 - Back")
    print("-"*40)
    flag = int(input(">> "))
    enable = 1
    # define path for saving onnx model
    # and tokenizer from huggingface
    if flag == 2:  
        onnx_path = Path("model/gpt_2/text_generator.onnx")
        onnx_path.parent.mkdir(exist_ok=True)
        pt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        dim = int(input("Select the max sequence length\n(max 1024, recommended 128)\n>>"))
        if dim < 5:
            dim = 5
        if dim > 1024:
            dim = 1024
    elif flag == 1:  
        onnx_path = Path("model/gpt_neo/text_generator.onnx")
        onnx_path.parent.mkdir(exist_ok=True)
        pt_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
        tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')
        dim = int(input("Select the max sequence length\n(max 2048, recommended 128)\n>>"))
        if dim < 5:
            dim = 5
        if dim > 1024:
            dim = 2048
    else:
        menu_fun()
        enable = 0
    if enable == 1:
        # save max sequence length
        os.system('cls')
        draw_sprite(1)
        print("Conversion ...\n")
        with open(onnx_path.with_suffix(".txt"), 'w') as f:
            f.write(f"{dim}")
        f.close()
        # define path for saving openvino model
        model_path = onnx_path.with_suffix(".xml")
        # get model onnx config function for output feature format casual-lm
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(pt_model, feature='causal-lm')
        # fill onnx config based on pytorch model config
        onnx_config = model_onnx_config(pt_model.config)
        # convert model to onnx
        onnx_inputs, onnx_outputs = export(preprocessor=tokenizer,model=pt_model,config=onnx_config,opset=onnx_config.default_onnx_opset,output=onnx_path)
        # convert model to openvino
        ov_model = mo.convert_model(onnx_path, compress_to_fp16=True, input=f"input_ids[1,1..{dim}],attention_mask[1,1..{dim}]")
        # serialize openvino model
        serialize(ov_model, str(model_path))

# INFERENCE FUNCTION
# inference with pt-2 and gpt-ne IR models
def inference_fun():
    os.system('cls')
    draw_sprite(2)
    print(" "*(20-round(9/2)),"INFERENCE")
    print("-"*40)
    print("Select the Model:\n 1 - GPT-Neo\n 2 - GPT-2\n 3 - Back\nUse 'q' in the chat to return to MENU")
    print("-"*40)
    flag = int(input(">> "))
    enable = 1
    # set path of IR models
    # and tokenizer from huggingface
    if flag == 2:  
        model_name.value = "GPT-2"
        model_path = Path("model/gpt_2/text_generator.xml")
        pt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif flag == 1:  
        model_name.value = "GPT-Neo"
        model_path = Path("model/gpt_neo/text_generator.xml")
        pt_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
        tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')
    else:
        menu_fun()
        enable = 0
    if enable == 1:
        # define max sequence length
        try:
            with open(model_path.with_suffix(".txt"), 'r') as f:
                max_dim = int(f.read())
            f.close()
        except:
            print("Missing Files")
            print("Exiting the program ...")
            sys.exit(0)
        dim = int(input(f"Select the max sequence length\n(max {max_dim})\n>>"))
        if dim > max_dim:
            dim = max_dim
        # define target device
        device = input("Choose target device\n(CPU, GPU, FPGA, HETERO)\n>>")
        os.system('cls')
        draw_sprite(1)

        # PRE-PROCESSING
        # define tokenization
        # this function converts text to tokens
        def tokenize(text: str) -> Tuple[List[int], List[int]]:
            """
            tokenize input text using GPT2 tokenizer

            Parameters:
              text, str - input text
            Returns:
              input_ids - np.array with input token ids
              attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
            """
            inputs = tokenizer(text, return_tensors="np")
            return inputs["input_ids"], inputs["attention_mask"]
        # define eos_token which means that generation is finished
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.decode(eos_token_id)

        # define softmax layer
        # this function is used to convert top-k logits into a probability distribution
        def softmax(x : np.array) -> np.array:
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            summation = e_x.sum(axis=-1, keepdims=True)
            return e_x / summation
            
        # set min sequence length
        # this function reduce the probability of the eos token occurring
        # this continues the process of generating the next words
        def process_logits(cur_length: int, scores: np.array, eos_token_id : int, min_length : int = 0) -> np.array:
            """
            Reduce probability for padded indices.

            Parameters:
              cur_length: Current length of input sequence.
              scores: Model output logits.
              eos_token_id: Index of end of string token in model vocab.
              min_length: Minimum length for applying postprocessing.

            Returns:
              Processed logits with reduced probability for padded indices.
            """
            if cur_length < min_length:
                scores[:, eos_token_id] = -float("inf")
            return scores

        # top-k sampling
        # the K most likely next words are filtered and
        # the probability mass is redistribute among only those K next words
        def get_top_k_logits(scores : np.array, top_k : int) -> np.array:
            """
            Perform top-k sampling on the logits scores.

            Parameters:
              scores: np.array, model output logits.
              top_k: int, number of elements with the highest probability to select.

            Returns:
              np.array, shape (batch_size, sequence_length, vocab_size),
                filtered logits scores where only the top-k elements with the highest
                probability are kept and the rest are replaced with -inf
            """
            filter_value = -float("inf")
            top_k = min(max(top_k, 1), scores.shape[-1])
            top_k_scores = -np.sort(-scores)[:, :top_k]
            indices_to_remove = scores < np.min(top_k_scores)
            filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                         fill_value=filter_value).filled()
            return filtred_scores


        # main processing function
        # generating the predicted sequence
        def generate_sequence(input_ids : List[int], attention_mask : List[int], max_sequence_length : int = 128,
                              eos_token_id : int = eos_token_id, dynamic_shapes : bool = True) -> List[int]:
            """
            Generates a sequence of tokens using a pre-trained language model.

            Parameters:
              input_ids: np.array, tokenized input ids for model
              attention_mask: np.array, attention mask for model
              max_sequence_length: int, maximum sequence length for stopping iteration
              eos_token_id: int, index of the end-of-sequence token in the model's vocabulary
              dynamic_shapes: bool, whether to use dynamic shapes for inference or pad model input to max_sequence_length

            Returns:
              np.array, the predicted sequence of token ids
            """
            
            while True:
                cur_input_len = len(input_ids[0])
                if not dynamic_shapes:
                    pad_len = max_sequence_length - cur_input_len
                    model_input_ids = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
                    model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
                else:
                    model_input_ids = input_ids
                    model_input_attention_mask = attention_mask
                outputs = compiled_model({"input_ids": model_input_ids, "attention_mask": model_input_attention_mask})[output_key]
                next_token_logits = outputs[:, cur_input_len - 1, :]
                # pre-process distribution
                next_token_scores = process_logits(cur_input_len,
                                                   next_token_logits, eos_token_id)
                top_k = 20
                next_token_scores = get_top_k_logits(next_token_scores, top_k)
                # get next token id
                probs = softmax(next_token_scores)
                next_tokens = np.random.choice(probs.shape[-1], 1,
                                               p=probs[0], replace=True)
                # break the loop if max length or end of text token is reached
                if cur_input_len == max_sequence_length or next_tokens[0] == eos_token_id:
                    break
                else:
                    input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                    attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)
            return input_ids

        # LOAD THE MODEL
        os.system('cls')
        # initialize openvino core
        core = Core()
        # read the model and corresponding weights from file
        model = core.read_model(model_path)
        # compile the model for CPU devices
        compiled_model = core.compile_model(model=model, device_name=device)
        # get output tensors
        output_key = compiled_model.output(0)

        # INFERENCE
        while(True):
            # inference with gpt-2 or gpt-neo
            text = input(Fore.WHITE +"\n[You]: ")
            if text == "q":
                break
            else:
                print(Fore.YELLOW + f"\n{model_name.value} is writing ...", end = "\r")
                input_ids, attention_mask = tokenize(text)
                start = time.perf_counter()
                output_ids = generate_sequence(input_ids, attention_mask, dim)
                end = time.perf_counter()
                output_text = " "
                # Convert IDs to words and make the sentence from it
                for i in output_ids[0]:
                    output_text += tokenizer.batch_decode([i])[0]
                print(f"Generation took {end - start:.3f} s")
                #print(f"Input Text:  {text}")
                #print()
                print(Fore.GREEN + f"\n[{model_name.value}]: {output_text}")


while(True):
    menu_fun()
    






    
