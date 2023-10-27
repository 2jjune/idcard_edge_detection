import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
# from tensorflow.keras.prerprocessing.image import ImageDataGenerator
# from tensorflow import keras
import edge_detection_model
from PIL import Image
from imutils.perspective import four_point_transform
import imutils
import math
from tensorflow.keras.preprocessing.image import array_to_img

img_size = 256

model = edge_detection_model.build_resnet50_unet((img_size, img_size, 3))
model.load_weights('./checkpoint/best.h5')
model.summary()
base_dir = 'C:/Users/dlwld/PycharmProjects/edge_detection/image_dataset_for_edge_detection/imgs/val/'
dataset = os.listdir(base_dir)


def get_area(findCnt):
    left_up = findCnt[0]
    left_down = findCnt[1]
    right_up = findCnt[2]
    right_down = findCnt[3]

    o1 = math.atan((left_up[1] - left_down[1]) / (left_up[0] - left_down[0]))
    o2 = math.atan((right_down[1] - left_down[1]) / (right_down[0] - left_down[0]))
    angle1 = abs((o1 - o2) * 180 / math.pi)

    o3 = math.atan((left_up[1] - right_up[1]) / (left_up[0] - right_up[0]))
    o4 = math.atan((right_down[1] - right_up[1]) / (right_down[0] - right_up[0]))
    angle2 = abs((o3 - o4) * 180 / math.pi)

    area = (0.5 * math.sqrt((left_up[0] - left_down[0]) ** 2 + (left_up[1] - left_down[1]) ** 2) * math.sqrt(
        (right_down[0] - left_down[0]) ** 2 + (right_down[1] - left_down[1]) ** 2) * math.sin(math.radians(angle1))) \
           + (0.5 * math.sqrt((left_up[0] - right_up[0]) ** 2 + (left_up[1] - right_up[1]) ** 2) * math.sqrt(
        (right_down[0] - right_up[0]) ** 2 + (right_down[1] - right_up[1]) ** 2) * math.sin(math.radians(angle2)))

    return area


def make_scan_image(base_org_image, image, min_threshold=180, max_threshold=220):
    org_image = image.copy()
    # image = imutils.resize(image, height=width, width=width)
    image = cv2.resize(image, (img_size,img_size))
    ratio = org_image.shape[1] / float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, min_threshold, max_threshold)

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None
    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break
    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:
        return 0
        # raise Exception(("Could not find outline."))

    area = get_area(findCnt.reshape(4,2))
    print(area)


    if area > (image.shape[0]**2)*0.8 or area < (image.shape[0]**2)*0.2 or np.isnan(area) == True:
        return 0
    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    findCnt = findCnt.reshape(4,2)
    for i in range(4):
        findCnt[i][0] *= (base_org_image.shape[1]/img_size)
        findCnt[i][1] *= (base_org_image.shape[0]/img_size)
    print(findCnt)
    transform_image = four_point_transform(base_org_image, findCnt)
    transform_image = cv2.resize(transform_image, (img_size, img_size))

    return transform_image

for img in dataset:
    print(img)
    org_image = cv2.imread(base_dir + img)
    image = cv2.resize(org_image,(img_size, img_size))
    image = np.asarray(image)
    image = image/255.
    image = np.expand_dims(image, axis=0)

    pred_mask = model.predict(image)

    pred_mask = np.squeeze(pred_mask, axis=0)
    pred_mask2 = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 1))

    for i in range(pred_mask.shape[0]):
        for j in range(pred_mask.shape[1]):
            if pred_mask[i][j]>0.5:
                pred_mask2[i][j]=255
    save_img = array_to_img(pred_mask2)
    save_img.save('./segment_result/{}'.format(img))

    save_img = np.array(save_img)
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

    receipt_image = make_scan_image(org_image, save_img, min_threshold=25, max_threshold=70)
    if type(receipt_image) == int:
        continue
    else:
        cv2.imwrite('C:/Users/dlwld/PycharmProjects/edge_detection/{}'.format(img), receipt_image)


