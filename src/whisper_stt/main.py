from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import aiofiles
from pydub import AudioSegment
import numpy as np
import io


import whisper

model = whisper.load_model("tiny.en")

def read_audio_file(file_path):
    # Use pydub to read the audio file and convert it to a float32 NumPy array
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples

# API setup
app = FastAPI()

class TranscribeInput(BaseModel):
    speech_file_path: str

@app.get("/")
async def home():
    return 'Success!'


@app.post("/transcribe")
# async def transcribe(audio_file: UploadFile = File(...)):
#     file_path = f"/user_data/tmp/{audio_file.filename}"
#     async with aiofiles.open(file_path, 'wb') as out_file:
#         content = await audio_file.read()
#         await out_file.write(content)  

async def transcribe(args: TranscribeInput):
    file_path = args.speech_file_path
    result = model.transcribe(file_path)
    
    return {"transcription": result["text"]}
    
    
# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)