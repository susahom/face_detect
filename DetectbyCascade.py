import numpy as np
import cv2

# Webカメラから入力
cap = cv2.VideoCapture(0)

# カスケードファイルを指定して、検出器を作成
face_cascade = cv2.CascadeClassifier("opencv-4.1.1/data/haarcascades/haarcascade_frontalface_alt2.xml")

print('O1K')

while True:
    # 画像を取得
    ret, img = cap.read()
    print('O2K')
    # グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=20)
    # 四角で囲む
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print('O3K')

    # imgという名前で表示
    cv2.imshow('img', img)

    # qキーが押されたら終了
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
