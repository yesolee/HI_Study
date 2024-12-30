# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite2.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3) # 3등까지 출력
classifier = vision.ImageClassifier.create_from_options(options)

# 웹서버보다 추론기 객체를 먼저 만들어 놓아야 에러가 안남

from fastapi import FastAPI, UploadFile

app = FastAPI()

import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # STEP 3: Load the input image.
    # open file => binary
    # http file => text (text로 인코딩 되서 정보가 전송됨) 
    contents = await file.read()
    # => 이걸 다시 binary로 바꿔줘야함
    nparr = np.fromstring(contents, np.uint8)
    # 이 binary는 jpg로 압축된 값이므로, 비트맵([256,256,256])으로 바꿔줘야 하므로 decode가 필요
    # cv_mat = cv2.imread(input_file) # imread = file open + image decode (압축된 jpg를 비트맵으로 바꿔줌)
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 메모리에 있는걸 디코드만 해주면 비트맵으로 바꿔줌
    
    # mediapipe를 사용하기 위해 요구하는 포맷으로 바꿔줌(insight를 쓸땐 안씀)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image) # forward(), inference(), get() 등 추론하는 함수 이름 다양함

    # STEP 5: Process the classification result. In this case, visualize it. 
    top_category = classification_result.classifications[0].categories[0] # 1등만 출력
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result": result}