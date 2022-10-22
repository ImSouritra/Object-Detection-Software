import cv2

# OpenCV DNN
net = cv2.dnn.readNet("dnn_model-220107-114215/dnn_model/yolov4-tiny.weights",
                      "dnn_model-220107-114215/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# load class lists

class_lists = []

with open("dnn_model-220107-114215/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_lists.append(class_name.split("\n")[0])

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)





while True:
    # Get Frames
    ret, frame = cap.read()



    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name  = class_lists[class_id]

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), thickness=3)



    cv2.imshow("FRAME", frame)
    key  = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()