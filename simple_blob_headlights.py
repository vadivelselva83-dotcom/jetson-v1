# file: simple_blob_headlights.py
import cv2, numpy as np

cap = cv2.VideoCapture(0)  # or use the same GStreamer pipeline as above
while True:
    ret, frame = cap.read();  if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # Top-hat to highlight bright spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # Adaptive threshold
    thr = cv2.threshold(tophat, 220, 255, cv2.THRESH_BINARY)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 30:  # reject tiny specks
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Headlight blob detector", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break
cap.release(); cv2.destroyAllWindows()
