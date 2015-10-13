# -*- coding:utf-8 -*-
import numpy
import cv2
def calc(detector_name, extractor_name, matcher_name, face_image,f):
    
    # キーポイントの検出
    detector = cv2.FeatureDetector_create(detector_name)
    keypoints1 = detector.detect(face_image)
    #print 'in:kp:',len(keypoints1)
    # 画像データの特徴量
    descripter = cv2.DescriptorExtractor_create(extractor_name)
    k1,d1 = descripter.compute(face_image, keypoints1)
    # matcher準備
    matcher = cv2.DescriptorMatcher_create(matcher_name)
    
    min_mean = 100000
    
    # テスト画像読み込み
    for i in xrange(1, 9):
        
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
        
        #min_mean = min(min(dist), min_mean)
        min_mean = min((sum(dist) / len(dist)), min_mean)
        
        #print 'distance: min: %.3f' % min(dist)
        #print 'distance: mean: %.3f' % (sum(dist) / len(dist))
        #print 'distance: max: %.3f' % max(dist)
     
        # threshold: half the mean
        #thres_dist = (sum(dist) / len(dist)) * 0.5
     
        # keep only the reasonable matches
        #sel_matches = [m for m in matches if m.distance < thres_dist]
     
        #print '#selected matches:', len(sel_matches)
    
    f.write('%d\t%s\t%s\t%s\t%d\n' % (len(keypoints1), detector_name, extractor_name, matcher_name, min_mean))
    print len(keypoints1), detector_name, extractor_name, matcher_name, min_mean
    

print('start')
cascade_path = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
filename = './images/nozomi-sasaki/01.jpg'
#ファイル読み込み
image = cv2.imread(filename)
print('to gray scale')
#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)
#物体認識（顔認識）の実行
print('顔認識')
facerect = cascade.detectMultiScale(
                image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
if len(facerect) <= 0:
    print u'not found!'
rect = facerect[0]
for r in facerect:
    if rect[2] < r[2]:
        rect = r
    
print('hoge')
x = rect[0]
y = rect[1]
w = rect[2]
h = rect[3]
# img[y: y + h, x: x + w] 
print('fuga')
face_image = image_gray[y:y+h, x:x+w]
cv2.imwrite("detected.jpg", face_image)
detectors = ['FAST','ORB','BRISK','MSER','GFTT','HARRIS','Dense']
extractors = ['ORB','BRISK','BRIEF','FREAK']
matchers = ['BruteForce','BruteForce-L1','BruteForce-SL2','BruteForce-Hamming','BruteForce-Hamming(2)']
f = open('result4.txt', 'w')
print('piyo')
for detector in detectors:
    for extractor in extractors:
        for matcher in matchers:
            calc(detector, extractor, matcher, face_image, f)
f.close()
