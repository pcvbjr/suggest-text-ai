import json
import os
import requests
import aiofiles

url = os.environ.get('WHISPER_API_URL')

async def stt(url: str, file_path: str):
    # async with aiofiles.open(file_path, 'rb') as f:
    #     audio_data = await f.read()
    response = requests.post(url+'/transcribe', json={"speech_file_path": file_path})
    response.raise_for_status()
    return response.json().get('transcription')
    
    
    
    # response = requests.post(url+'/transcribe', json={'speech_file_path': speech_file})
    # transcription = json.loads(response.text).get('transcription', '')
    # return transcription