import torch
print(torch.cuda.is_available())
import whisper

voice_model = whisper.load_model("large-v3",device="cuda")
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
# from google.colab import userdata


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


token = 'AIzaSyAlfizEQTTHDLx-xzYZZ9Sauu2mqIct2XQ'
genai.configure(api_key = token)


for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)


model = genai.GenerativeModel('gemini-pro')

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
from urllib.parse import urlencode
import uuid
import os
from elevenlabs import  play, save
from elevenlabs.client import ElevenLabs
import asyncio
import io
from fastapi.responses import StreamingResponse
import json
import joblib
import spacy

client = ElevenLabs(
  api_key="9cae23f11ee9d1176ed0a3c46d8fe7bf", # Defaults to ELEVEN_API_KEY
)

nlp = spacy.load("en_core_web_sm")
app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

class TranscriptionRequest(BaseModel):
    audio: UploadFile

class QueryRequest(BaseModel):
    query: str

class TTSNER(BaseModel):
    text: str
    emotion: str = "Cheerful & Professional"

@app.post("/transcription")
async def transcribe_and_classify_audio(audio: UploadFile = File(...)):
    audio_path = f"{uuid.uuid4()}.wav"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    result = voice_model.transcribe(whisper.pad_or_trim(whisper.load_audio(audio_path)))
    text = result["text"]
    source_text = text
    src_lang = result["language"]
    print(text)
    os.remove(audio_path)
    res = model.generate_content(f"""

  [CONTEXT]
  You are an AI classification model that needs to classify if the user query pertains to Information Retrieval or not.

  [INSTRUCTIONS]
  You will be provided a user query, and you must indicate whether the query requires the extraction of information such as facts, summaries, news articles, etc.
  The query shall also be considered Information Retrieval if it asks to perform basic or complex calculations.


  [EXAMPLES]
  1. User query: "How's the weather today?"
    Type: Information Retrieval

  2. User query: "Set a reminder for tomorrow's meeting at 10 AM."
    Type: None

  3. User query: "What's your favorite movie?"
    Type: Information Retrieval

  4. User query: "Find the nearest coffee shop."
    Type: None

  5. User query: "What is 30% of 50?"
      Type: Information Retrieval

  [USER QUERY]
  {text}

  [ANSWER]
  """)
    print(res.text)

    if res.text == 'Information Retrieval':
        response = await info_retrieval_endpoint(text, src_lang, source_text)
    elif res.text == 'None':
        response = await taskbased_endpoint(text, src_lang, source_text)
    else:
        response = {"error": "Unexpected classification result"}

    return response

@app.post("/file")
async def email_endpoint(file_audio: UploadFile = File(...)):
    audio_path = f"{uuid.uuid4()}.wav"
    with open(audio_path, "wb") as f:
        f.write(await file_audio.read())

    result = voice_model.transcribe(whisper.pad_or_trim(whisper.load_audio(audio_path)))
    text = result["text"]
    source_text = text
    src_lang = result["language"]
    print(text)
    os.remove(audio_path)
    res = model.generate_content(f"""
    [INSTRUCTIONS]
    You are an AI model which takes in a request to upload/push/pull different types of files. Your task is to return both the MEDIUM of updation of files as well as the FILE NAME
    with its extension.
    [EXAMPLES]
    1. User Query: Please upload my file phone.js to GitHub.
    Output: ['upload','phone.js']

    2. User Query: Please push my file config.py to GitHub.
    Output: ['push','config.py']

    3. User Query: Kindly pull from this repository -  Innomer
    Output: ['pull','Innomer']

    [QUERY]
    {text}
                                 
    [ANSWER]                                 
    """)

    res.text.strip().split(", ")


async def info_retrieval_endpoint(query: str, src_lang: str, source_text: str):
    res = model.generate_content(f'You are a friendly AI model. Give a succinct and pleasant answer to this question - {query}')
    print(res.text)
    return {"message": res.text, "src_lang": src_lang, "src": source_text}

async def taskbased_endpoint(query: str, src_lang: str, source_text: str):
    pipelineTree = joblib.load('decision_tree_model.pkl')
    pred = pipelineTree.predict([query])
    # Now you can use the loaded model for prediction
    print("Predicted Label:", pred)
    return {"predicted": pred[0], "src_lang": src_lang, "src": source_text}

email_lookup = {
    "Varun": "varunvis2903@gmail.com",
    "Varun Viswanath": "varunvis2903@gmail.com",
    "Aman": "aman2003nambisan@gmail.com",
    "Aman Nambisan": "aman2003nambisan@gmail.com",
    "Aman Nabisan": "aman2003nambisan@gmail.com",
    "Mann": "bhanushalimann@gmail.com",
    "Man": "bhanushalimann@gmail.com",
    "Mann Bhanushali": "bhanushalimann@gmail.com",
    "John Doe": "johndoe123@gmail.com",
    "Jake Smith": "smithj@gmail.com"
}

@app.post("/email")
async def email_endpoint(query_req : QueryRequest= Body(...)):
    text = query_req.query
    doc = nlp(text)
    names = [entity.text for entity in doc.ents if entity.label_ == "PERSON"]
    res = model.generate_content(f"""Give a brief and succinct email SUBJECT to this query.
    Return the SUBJECT as a string. Do NOT perform any formatting.
    {text}""")

    mailto = None
    cc = []

    # Lookup email addresses for names in the query
    for name in names:
        if name in email_lookup:
            # If name found in lookup, assign the corresponding email address to mailto or cc
            email = email_lookup[name]
            if mailto is None:
                mailto = email
            else:
                cc.append(email)

    # Return response with mailto and cc
    return {"mailto": mailto,"subject":res.text, "cc": cc}

from googlesearch import search
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import warnings
warnings.filterwarnings('ignore')
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless=new") 

# def returnLinks(query):
#    # YT part
    

# print(returnLinks("How to tie my shoelaces"))

@app.post('/web-search')
async def web_search_endpoint(query_req: QueryRequest = Body(...)):
    query = query_req.query
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"https://www.youtube.com/results?search_query={query}")
    wait = WebDriverWait(driver, 2)
    user_data = driver.find_elements(By.XPATH, '//a[@id="video-title"]')
    linksYT = []
    for i in user_data:
        attr = i.get_attribute("href")
        if 'shorts' not in i.get_attribute("href"):
            linksYT.append(attr[attr.find("=") + 1:attr.find("&")])

    linksGoogle = []
    # google part
    # search_results = search(
    #     f"https://www.bing.com/news/search?q={query}", num_results = 5)
    # for _, link in enumerate(search_results, start = 1):
    #     linksGoogle.append(link)

    return {"youtube" : linksYT, "news" : linksGoogle}



@app.post("/labs-tts/")
async def labs_tts(request: TTSNER = Body(...)):#,token: str = Header(...)):
    try:
      #decoded_token = decode_token(token)
      out = f"{uuid.uuid4()}.mp3"
      async def remove():
          loop = asyncio.get_event_loop()
          await loop.run_in_executor(None, lambda: os.remove(out))
      #out=ttsclient.predict(request.text,"Ryan",fn_index=0)
      audio = client.generate(
        text=request.text,
        voice="Sarah",
        model="eleven_multilingual_v2"
      )
      save(audio,out)
      return FileResponse(out,headers={"Content-Disposition":f"attachment; filename={out}"},background=remove)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
import nest_asyncio
from pyngrok import ngrok
import uvicorn



ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port = 8000)