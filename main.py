import os
import json

import requests
from inflection import singularize
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# URL = ""
headers = {
    "Content-Type": "application/json"
}


## Preprocessing ##

def preprocess(img):
    img = cv2.resize(img, None, fx=3.3, fy=3.3, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


## build a dictionary ##
def add_to_dictionary(page: str, doc: dict):
    return doc


## Save to Firestore ##
def request_to_server(book_title: str, doc: dict):
    # 1. send to crayon server
    # 2. crayon server will save it to Firestore.
    data = {"book_title": book_title}
    doc = json.dumps(doc)

    print(f"sending {doc} to server......")

    response = requests.post(url=URL + f"?book_title={book_title}", headers=headers, data=doc)
    print(response)
    print(response.text)


## Main ##

books_path = "./books"
results_path = "./results"
book_title = "Come Over"

for book_title in os.listdir(books_path):
    if book_title == ".DS_Store":
        continue
    img_path = f"{books_path}/{book_title}"
    result_path = f"{results_path}/{book_title}"
    
    # firestore
    doc = {}
    print(f"##### Processing {book_title} .... #####")

    for j in range(1, len(os.listdir(img_path))):
        
        if os.path.isfile(f"{img_path}/{j}.JPG"):
            img = cv2.imread(f"{img_path}/{j}.JPG")
        else:
            continue

        img = preprocess(img)

        # firestore
        bounding_boxes = []

        # cv2.imshow('img', gray_img)
        # cv2.waitKey(0)

        # dic = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)
        dic = pytesseract.image_to_data(img, lang='eng', config='--oem 1', output_type=Output.DICT)
        
        print(f"##### Working on page {j} .... #####")

        px, py = img.shape
        px *= 2

        n_boxes = len(dic['text'])
        for i in range(n_boxes):
            if int(dic['conf'][i]) > 60:
                (x, y, w, h) = (dic['left'][i], dic['top'][i], dic['width'][i], dic['height'][i])
                word = dic['text'][i]

                if word is not ('' or ' '):
                    
                    # firestore
                    box_info = {}

                    word = word.lower()

                    if word[-1] in [".", ",", "!", "?"]:
                        word = word[:-1]
                    
                    # print(f"{word} \t {singularize(word)}")
                    word = singularize(word)

                    x_start_ratio = x / px
                    y_start_ratio = y / py

                    x_end_ratio = (x + w) / px
                    y_end_ratio = (y + h) / py

                    if not j % 2: # 오른쪽 페이지.
                        x_start_ratio += 0.5
                        x_end_ratio += 0.5

                    # firestore
                    box_info[word] = f"[{x_start_ratio}, {y_start_ratio}, {x_end_ratio}, {y_end_ratio}]"
                    bounding_boxes.append(box_info)

                    # print(f"x: {x_start_ratio} \t y: {y_start_ratio}")
                    # print(f"x + w: {x_end_ratio} \t y + h : {y_end_ratio}")
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)   

        # firestore
        if not j % 2:
            doc[j-1] += bounding_boxes
        else:
            doc[j] = bounding_boxes
        # print(doc)

        # print(img.shape)
        # cv2.imshow('img', img)
        
    # Firestore
    # print(doc)
    request_to_server(book_title, doc)
    # cv2.waitKey(0)

    # cv2.imshow('img', img)
    # cv2.imwrite(f"{result_path}/{j}.JPG", img)