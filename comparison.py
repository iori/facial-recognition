# -*- coding:utf-8 -*-
import numpy
import cv2
def calc(detector_name, extractor_name, matcher_name, face_image):
    
    # キーポイントの検出
    detector = cv2.FeatureDetector_create(detector_name)
    keypoints1 = detector.detect(face_image)
    # 画像データの特徴量
    descripter = cv2.DescriptorExtractor_create(extractor_name)
    k1,d1 = descripter.compute(face_image, keypoints1)
    # matcher準備
    matcher = cv2.DescriptorMatcher_create(matcher_name)
    
    min_dist = 100000
    
    # テスト画像読み込み
    for i in xrange(1, 12):
        
        # 画像はグレースケール変換済
        test_file = "./images/namie-amuro/face_%02d.jpg" % i
        test_image = cv2.imread(test_file)
        
        #print test_file
        
        # キーポイントの検出
        keypoints2 = detector.detect(test_image)
        #print len(keypoints2)
        k2,d2 = descripter.compute(test_image, keypoints2)
        # キーの一致度合いを調べる
        try:
            matches = matcher.match(d1, d2)
        except:
            continue
        #print '#matches:', len(matches)
        dist = [m.distance for m in matches]
        
        if len(dist) == 0:
            continue
        
        min_dist = min(min(dist), min_dist)
    
    return min_dist
    
cascade_path = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
for i in xrange(1, 13):
    filename = './images/pics/%02d.jpg' % i
    #ファイル読み込み
    image = cv2.imread(filename)
    #グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    #物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(
                    image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(facerect) <= 0:
        print u'not found!'
    # 適当に一番大きそうな画像を探す
    rect = facerect[0]
    for r in facerect:
        if rect[2] < r[2]:
            rect = r
        
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    # img[y: y + h, x: x + w] 
    face_image = image_gray[y:y+h, x:x+w]
    cv2.imwrite("detected_" + filename, face_image)
    resutl = calc('Dense', 'BRISK', 'BruteForce-Hamming', face_image)
    if resutl < 100:
        print filename, resutl, 'namie'
    else:
        print filename, resutl, 'not namie'
