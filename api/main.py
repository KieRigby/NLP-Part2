from fastapi import FastAPI, Request
import json
import numpy as np
import requests
from utils.utils import lemmatise, remove_stopwords, clean_lyrics, tokenize
from pydantic import BaseModel
from typing import List
import logging
from helpers import async_iterator_wrapper as aiwrap
from helpers import setup_logging, get_body
app = FastAPI()

setup_logging()
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_request(request: Request, call_next):
    logger.info(f'{request.method} {request.url}')
    req_body= await get_body(request)
    logger.info(f'Request Body: {req_body}')

    response = await call_next(request)
    logger.info(f'Status code: {response.status_code}')
    # Consuming FastAPI response and grabbing body here
    resp_body = [section async for section in response.__dict__['body_iterator']]
    # Repairing FastAPI response
    response.__setattr__('body_iterator', aiwrap(resp_body))
    # Formatting response body for logging
    try:
        resp_body = json.loads(resp_body[0].decode())
    except:
        resp_body = str(resp_body)
    logger.info(f'Response Body: {resp_body}')

    return response
    
class Lyrics(BaseModel):
    lyrics: List[str]

@app.post('/lyricclassifier/predict/')
def lyric_classifier(lyrics:Lyrics):
    SERVER_URL = 'http://tf_serving:8501/v1/models/lyrics_model:predict'

    preprocessed_lyrics =[lemmatise(remove_stopwords(clean_lyrics(lyric))) for lyric in lyrics.lyrics]
    input, empty_array, tokenizer_inference = tokenize(preprocessed_lyrics, [])
    data = json.dumps({"signature_name": "serving_default", "instances": input.tolist()})
    # Making POST request
    response = requests.post(SERVER_URL, data=data)
    predictions=response.json()['predictions']
    classification_labels=[]
    for i,prediction in enumerate(predictions):
        classifications=[]
        if prediction[0] > 0.5:
            classifications.append('alternative')
        if prediction[1] > 0.5:
            classifications.append('folk')
        if prediction[2] > 0.5:
            classifications.append('negative')
        if prediction[3] > 0.5:
            classifications.append('pop')
        if prediction[4] > 0.5:
            classifications.append('postive')
        if prediction[5] > 0.5:
            classifications.append('rap')
        if prediction[6] > 0.5:
            classifications.append('rock')

        classification_labels.append(classifications)

    return {'predictions': classification_labels}

    