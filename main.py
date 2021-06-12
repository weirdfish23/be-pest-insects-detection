from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
import io
import torch
import cv2
from PIL import Image
import numpy as np
from models_detection import make_detection, save_on_s3

app = FastAPI()


@app.get("/")
def read_root():
    return "El Psy Congroo!"


@app.post("/pest_detection")
def pest_detection(file: UploadFile = File(...)):
    filename = file.filename
    img = Image.open(io.BytesIO(file.file.read()))
    print('Filename::', filename)
    print('Img. size::', img.size)
    result_img, result_df = make_detection(img)
    print('Result Img. size::', result_img.size)
    print('Result DF. len::', len(result_df))
    url_result_img, url_result_csv = save_on_s3(result_img, result_df, filename)

    return {'url_result_img':url_result_img, 'url_result_df':url_result_csv}


