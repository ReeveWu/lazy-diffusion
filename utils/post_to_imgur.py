import requests
import base64
import numpy as np
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID')

def img_post(image):
    # 將numpy陣列轉換為圖片格式
    # image = cv2.imread('Avater.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.png', image)

    # 將圖片轉換為base64編碼的字串
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # 上傳圖片到imgur
    response = requests.post(
        'https://api.imgur.com/3/image',
        headers={'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'},
        data={'image': image_base64}
    )

    # 解析回傳的JSON資料，取得圖片網址
    img_link = response.json()['data']['link']

    return img_link