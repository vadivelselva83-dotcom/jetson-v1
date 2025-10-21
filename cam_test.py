# file: cam_test.py
import cv2

# 1920x1080 @ 30fps, BGR frames to OpenCV
def gstreamer_pipeline(width=1920, height=1080, fps=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit("Could not open IMX219 via GStreamer. Check ribbon cable & drivers.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "IMX219 1080p @ 30fps", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("IMX219 Preview", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
