# -*- coding:utf-8 -*-
import numpy
import cv2
cascade_path = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
for i in xrange(1, 12):
    path = "./images/namie-amuro/"
    filename = "%02d.jpg" % i
    filepath = path + filename
    
    #ファイル読み込み
    image = cv2.imread(filepath)
    #グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    #物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(
                    image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    
    print filepath, len(facerect)
    
    if len(facerect) <= 0:
        continue
    
    rect = facerect[0]
    for r in facerect:
        if rect[2] < r[2]:
            rect = r
        
    
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    
    # img[y: y + h, x: x + w] 
    new_image_name = 'face_' + filename
    new_image = cv2.imwrite(path + new_image_name, image_gray[y:y+h, x:x+w])
    print(image_gray)
    print(new_image)

