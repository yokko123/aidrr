import torch
import cv2

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import check_img_size
from yolov5.utils.torch_utils import select_device, time_sync

model = torch.hub.load('ultralytics/yolov5', 'custom', path='fire.pt')


cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
attempt_download('deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
device = select_device('0')
#load model
imgsz = 640
device = select_device(device)
model = DetectMultiBackend('fire.pt',device=device,dnn=True)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names

model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))
dt, seen = [0.0, 0.0, 0.0], 0
cap = cv2.VideoCapture('test1.mp4')

#ret, frame2 = cap.read()
while cap.isOpened():
    ret, img = cap.read()
    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t2 = time_sync()
    dt[0] += t2-t1

    pred = model(img, augment=True, visualize =True)

    t3 = time_sync()
    dt[1] += t2-t2

    cv2.imshow('output', img)









    # diff = cv2.absdiff(frame1,frame2)
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # im_bw = cv2.Canny(blur, 10, 90)
    # _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # dilated = cv2.dilate(thresh, None, iterations=3)
    # contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # result = model(frame1)
    # for contour in contours:
    #     (x,y,w,h) = cv2.boundingRect(contour)
    #
    #     if cv2.contourArea(contour) < 500:
    #         continue
    #     cv2.rectangle(frame1,(x,y),(x+w,y+h), (0,255,0), 2)
    #     cv2.putText(frame1, "Status: {}".format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (0,0,255), 3)
    #
    # # cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    # cv2.imshow('output', frame1)
    # frame1 = frame2
    # ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
