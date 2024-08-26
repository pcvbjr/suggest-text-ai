import json
import os
import re
import string

import numpy as np
from openai import OpenAI
    
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL')
OPENAI_API_KEY = os.environ.get('OPEN_API_KEY')
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)



def llm_request(prompt: str, temperature: float):
    """
    Makes a request to the language model API with the specified parameters.
    
    Args:
        url_base (str): The base URL of the language model API.
        prompt (str): The prompt to generate text for.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 5.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.
        logprobs (float, optional): The log probability parameter for the language model. Defaults to 5.
    
    Returns:
        str: The generated text from the language model API.
    """
    model = client.models.list().data[0].id
    
    params ={
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": temperature,
        "logprobs": 5,
        "seed": 0,
        "best_of": 256,
        "n": 256,
    }
    response = client.completions.create(**params)
    return response.choices

def get_logprobs(response):
    # return response['logprobs']['top_logprobs']
    response_logprobs = list(set([(completion.text, completion.logprobs.token_logprobs[0]) for completion in response]))
    return response_logprobs

def create_word_completion_instruction(prompt):
    examples = [
        "Sentence: At the movie we should get some c\nWord: candy",
        "Sentence: I am pretty tir\nWord: tired",
        "Sentence: My favorite book series is Harry P\nWord: Potter",
        "Sentence: I am going to a concert. We are seeing the classic rock band the Roll\nWord: Rolling",
        "Sentence: I'll have chicken noodl\nWord: noodle",
    ]
    
    system_instruction = "Instruction: Predict the completed last word in the sentence. \n\n" + '\n\n'.join(examples) + '\n\nSentence: ' + prompt + '\nWord: '
    return system_instruction

def clean_token(token, prompt):
    printable = set(string.printable)
    SPACE_CH = 'Ġ'
    NEWLN_CH = 'Ċ'
    if re.match(r'^[\W_]+$', token.strip()) or re.match(r'\s*<.*>', token.strip()):
        return None
    token = token.replace(SPACE_CH, ' ').replace(NEWLN_CH, ' ')
    token = ''.join(filter(lambda x: x in printable, token))
    if prompt[-1] == ' ':
        token = token.strip()
    if len(token) == 0:
        return None
    return token.lower()

def llm_token_logprobs(prompt: str):

    # if prompt ends in space or punctuation, predict next word
    if prompt[-1] in string.punctuation or prompt[-1] == ' ':
        final_word = ''
        res = llm_request(prompt.strip(), 0.5)
        lp = get_logprobs(res)
        response_logprobs = []
        for tok, tok_logprob in lp:
            tok = tok.strip()
            response_logprobs.append((tok, tok_logprob))
    
    # if prompt ends in word, complete the word
    else:
        completion_instruction = create_word_completion_instruction(prompt)
        final_word = prompt.split()[-1]
        res = llm_request(completion_instruction, 0.9)
        lp = get_logprobs(res)
        response_logprobs = []
        for tok, tok_logprob in lp:
            tok = tok.strip()
            if tok.startswith(final_word):
                response_logprobs.append((tok, tok_logprob))
                
    return response_logprobs, final_word

def llm_next_token_probs(prompt):
    
    top_k = None
    top_p = None
    p_threshold = None
    include_alphabet = True
    
    if include_alphabet:
        ALPHABET_LC = list(map(chr, range(97, 123))) + [' ']
        char_probs = {c: 0.0001 for c in ALPHABET_LC}
        if prompt == '':
            return [(1/len(char_probs), c) for c, c_prob in char_probs.items()]
    
    response_logprobs, final_word = llm_token_logprobs(prompt)
    
    token_probs = {}
    k_count = 0
    p_sum = 0
    add_tokens = True
    for tok, tok_logprob in response_logprobs:
        tok = clean_token(tok, prompt)
        if tok is None:
            continue
        
        tok_prob = np.exp(tok_logprob)
        k_count += 1
        p_sum += tok_prob
        if top_k is not None and k_count > top_k:
            add_tokens = False
        if top_p is not None and p_sum > top_p:
            add_tokens = False
        if p_threshold is not None and tok_prob < p_threshold:
            add_tokens = False
        
        if add_tokens:
            token_probs[tok] = tok_prob
        if include_alphabet:
            if tok.strip() and tok.strip()[0] in char_probs:
                try:
                    char_probs[tok.strip().replace(final_word, '')[0]] += tok_prob
                except:
                    pass
                
    if include_alphabet:    
        token_probs.update(char_probs)
    normalization_factor = sum([tok_prob for tok_prob in token_probs.values()])
    if normalization_factor == 0:
        token_probs_list = [(0, tok) for tok, tok_prob in token_probs.items()]
    else:
        token_probs_list = [(tok_prob/normalization_factor, tok) for tok, tok_prob in token_probs.items()]
        token_probs_list = sorted(token_probs_list, key=lambda x:(-x[0], x[1]))
    return token_probs_list
    
                
    
    
