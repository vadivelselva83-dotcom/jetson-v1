# file: detect_live.py
import cv2
import time
import numpy as np
import tensorflow as tf

MODEL_DIR = "outputs_retinanet/savedmodel_trt_fp16"
INPUT_SIZE = 640
SCORE_THRESH = 0.4
IOU_THRESH = 0.5

def gstreamer_pipeline(width=1920, height=1080, fps=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )

def preprocess(frame):
    # Optional: glare-suppression (CLAHE on V channel)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    frame = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    frame = cv2.GaussianBlur(frame, (3,3), 0)

    # Letterbox resize to INPUT_SIZE, keep aspect
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE/w, INPUT_SIZE/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    top = (INPUT_SIZE - nh)//2
    left = (INPUT_SIZE - nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    img = canvas.astype(np.float32) / 255.0
    return img, scale, left, top, (w, h)

def postprocess(preds, scale, left, top, orig_wh):
    # KerasCV RetinaNet SavedModel returns dict with 'boxes','classes','confidence' OR a nested structure
    # We try common signatures:
    if isinstance(preds, (list, tuple)):
        # [boxes, scores, classes, num_dets]
        boxes, scores, classes = preds[0], preds[1], preds[2]
    elif isinstance(preds, dict):
        boxes  = preds.get("boxes", preds.get("box_predictions", None))
        scores = preds.get("confidence", preds.get("scores", None))
        classes= preds.get("classes", None)
    else:
        raise RuntimeError("Unknown model outputs")

    boxes = boxes[0].numpy()  # [N,4] in xyxy relative to 640x640
    scores = scores[0].numpy()
    classes= classes[0].numpy().astype(int)

    # Remove padding & map back to original image coordinates
    ow, oh = orig_wh
    inv_scale = 1.0/scale
    # undo letterbox
    boxes[:, [0,2]] -= left
    boxes[:, [1,3]] -= top
    boxes *= inv_scale

    # clip
    boxes[:,0] = np.clip(boxes[:,0], 0, ow-1)
    boxes[:,1] = np.clip(boxes[:,1], 0, oh-1)
    boxes[:,2] = np.clip(boxes[:,2], 0, ow-1)
    boxes[:,3] = np.clip(boxes[:,3], 0, oh-1)

    keep = scores >= SCORE_THRESH
    return boxes[keep], scores[keep], classes[keep]

print("Loading TF-TRT model...")
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures.get("serving_default", None)
if infer is None:
    # Some exports use __call__
    infer = model

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit("Could not open camera. Check cabling / nvargus-daemon.")

fps_avg = 0.0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img, scale, left, top, orig_wh = preprocess(frame)
    inp = tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)

    t_in = time.time()
    preds = infer(inp)
    t_out = time.time()

    boxes, scores, classes = postprocess(preds, scale, left, top, orig_wh)

    for (x1,y1,x2,y2), sc in zip(boxes.astype(int), scores):
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"Headlight {sc:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    fps = 1.0 / max(1e-6, (t_out - t_in))
    fps_avg = 0.9*fps_avg + 0.1*fps
    cv2.putText(frame, f"FPS: {fps_avg:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Headlight detection (TF-TRT)", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
