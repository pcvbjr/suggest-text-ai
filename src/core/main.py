import json
import os
import warnings

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
from typing import List, Dict, Any
from pydantic import BaseModel
import uvicorn
import validators

from modules.llm_suggest import llm_next_token_probs
from modules.stt import stt
from modules.full_response import generate_full_responses

def word_or_char(token):
    if len(token) == 1:
        return 'char'
    elif len(token) > 1:
        return 'word'
    else:
        return ''

def get_suggested_words(input_string_stage, token_probs, top_k, p_threshold):
    '''Returns list of words in token_probs'''
    words = []
    for tok_prob, tok in token_probs:
        if word_or_char(tok) == 'word':
            words.append(tok)
    return words[:top_k]

def get_suggested_chars(token_probs, top_k, p_threshold):
    '''Returns list of top characters in token_probs'''
    min_tok_prob = 1e10
    for tok_prob, tok in token_probs:
        if tok_prob < min_tok_prob:
            min_tok_prob = tok_prob
    chars = []
    for tok_prob, tok in token_probs:
        if word_or_char(tok) == 'char':
            if (tok_prob < p_threshold and len(chars) >= 5) or len(chars) >= top_k:
                break
            if tok_prob <= min_tok_prob:
                break
            chars.append(tok.replace(' ', '[space]'))
    return chars


# Inference hyperparameters
word_top_k = 4
word_p_threshold = 0.01
char_top_k = 10
char_p_threshold = 0.001


# API setup
app = FastAPI()

class SetEnvInput(BaseModel):
    env_name: str
    env_value: str

class SuggestInput(BaseModel):
    text: str
    
class ConvoHistory(BaseModel):
    conversation_history: str
    
@app.get("/")
async def home():
    home = 'This is an API for suggest-text. See full docs at /docs.'
    return home

@app.get("/user_name")
async def get_user_name():
    user_name = os.environ.get('USER_NAME', '')
    if not user_name:
        print('USER_NAME is not set. Set with POST /set_env.')
    return user_name

@app.post("/set_env")
async def set_env(args: SetEnvInput):
    known_env_names = ['WHISPER_API_URL', 'OPENAI_BASE_URL', 'OPENAI_API_KEY']
    if args.env_name not in known_env_names:
        warnings.warn(f'{args.env_name} is not a known environment variable name. The known environment variable names are {",".join(known_env_names)}')
    if 'URL' in args.env_name and not validators.url(args.url_value):
        raise ValueError(f'{args.url_value} is not a valid URL.')
    os.environ[args.url_name] = args.url_value
    
@app.post("/suggest")
async def suggest_text(args: SuggestInput):
    token_probs = llm_next_token_probs(args.text) 
    suggested_words = get_suggested_words(args.text, token_probs, word_top_k, word_p_threshold)
    suggested_chars = get_suggested_chars(token_probs, char_top_k, char_p_threshold)
    return {"words": suggested_words, "chars": suggested_chars}


@app.post("/full-response")
async def full_response(audio_file: UploadFile = File(...), convo_history: str = Form(...)):

    file_path = f"/user_data/tmp/{audio_file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await audio_file.read()
        await out_file.write(content)  
    
        
    # Forward the file to the STT service
    speech_text = await stt(
        os.environ.get('WHISPER_API_URL'),
        file_path,
    )
    print('SPEECH:', speech_text)

    # Parse the conversation history
    conversation_history = json.loads(convo_history)
    
    conversation_history.append({'role': 'user', 'content': speech_text})

    # generate 3 different responses
    responses, adjectives = generate_full_responses(conversation_history)

    return {"responses": responses, "adjectives": adjectives, 'transcription': speech_text}





    

    
# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)