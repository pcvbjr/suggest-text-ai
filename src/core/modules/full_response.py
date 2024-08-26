import os
from typing import List, Dict

from openai import OpenAI


OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

USER_NAME  = os.environ.get('USER_NAME')
PROMPT_PREFIX = f"Respond as if you are a person named {USER_NAME}, engaged in live conversation with the user."
PROMPT_VARIABLE = "Your reponse should be short (max 100 characters), and generally [ADJ]."
SYSTEM_PROMPT = PROMPT_PREFIX + PROMPT_VARIABLE

def llm_request(conversation_history: List[Dict[str, str]], system_prompt: str, temperature: float):
    # create request params
    messages = [{'role': 'system', 'content': system_prompt}] + conversation_history
    print(OPENAI_BASE_URL, OPENAI_API_KEY)
    print('MESSAGES\n', messages)
    model = client.models.list().data[0].id
    print(model)
    params = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
    }
    
    # make request
    completion = client.chat.completions.create(**params)
    
    # return choices
    return completion.choices[0].message.content

def generate_instructions(conversation_history: List[Dict[str, str]]):
    sys_prompt = "Your job is to return three very adjectives to describe how a person in this conversation could respond to the user. The first adjective should be positive, the second should be negative, and the third should be neutral or a wild card. Respond only with the three adjectives, separated by commas (,)."
    
    # Use examples as conversation_history, instead of previous rounds of conversation
        # # example template
        # {'role': 'user', 'content': ""},
        # {'role': 'assistant', 'content': ''},
    examples = [
        # example 1
        {'role': 'user', 'content': "You outfit looks fantastic today!"},
        {'role': 'assistant', 'content': 'passionate,negative,unaffected'},
        # example 2
        {'role': 'user', 'content': "What do you think about the latest movie?"},
        {'role': 'assistant', 'content': 'positive,negative,neutral'},
        # example 3
        {'role': 'user', 'content': "I'm sorry for being late."},
        {'role': 'assistant', 'content': 'forgiving,offended,empathetic'},
        # example 4
        {'role': 'user', 'content': "I've always wanted to learn how to play the guitar."},
        {'role': 'assistant', 'content': 'supportive,disinterested,curious'},
    ]
    
    conversation_history = examples + [conversation_history[-1]]
    
    
    adjectives = llm_request(conversation_history, sys_prompt, temperature=0.7)
    try:
        adjectives = clean_llm_response(adjectives)
        adjectives = adjectives.split(',')
        assert len(adjectives) == 3
        for a in adjectives:
            assert isinstance(a, str)
            assert len(a) > 0
    except Exception as e:
        adjectives = ['positive', 'negative', 'neutral']
    return adjectives

def clean_llm_response(response):
    response = response.replace('*', '')
    return response

def generate_full_responses(conversation_history: List[Dict[str, str]]):
    responses = []
    adjectives = generate_instructions(conversation_history)
    system_prompts = [SYSTEM_PROMPT.replace("[ADJ]", adj) for adj in adjectives]
    for sys_prompt in system_prompts:
        response = llm_request(conversation_history, sys_prompt, temperature=0.2)
        response = clean_llm_response(response)
        responses.append(response)
    return responses, adjectives
        
    
    