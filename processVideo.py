import cv2
import numpy as np
import torch
import useModel
from sklearn.preprocessing import label_binarize


# 初始化一些固定的值，加快视频播放时的运算
emotion_dict={}
emotion_dict[0]=cv2.resize(cv2.imread("./qqemo/Angry.png", cv2.IMREAD_COLOR),(200,200))
emotion_dict[1]=cv2.resize(cv2.imread("./qqemo/Disgust.png", cv2.IMREAD_COLOR),(200,200))
emotion_dict[2]=cv2.resize(cv2.imread("./qqemo/Fear.png", cv2.IMREAD_COLOR),(200,200))
emotion_dict[3]=cv2.resize(cv2.imread("./qqemo/Happy.png", cv2.IMREAD_COLOR),(200,200))
emotion_dict[4]=cv2.resize(cv2.imread("./qqemo/Sad.png", cv2.IMREAD_COLOR),(200,200))
emotion_dict[5]=cv2.resize(cv2.imread("./qqemo/Surprise.png", cv2.IMREAD_COLOR),(200,200))
emotion_dict[6]=cv2.resize(cv2.imread("./qqemo/Neutral.png", cv2.IMREAD_COLOR),(200,200))

# 实例化级联分类器,加载分类器
face_path = 'haarcascade_frontalface_default.xml'
face_cas = cv2.CascadeClassifier(face_path)
face_cas.load(face_path)
# 加载模型
net=useModel.CNN()
net.load_state_dict(torch.load("./net_parameter63.pkl"))

# 显示的emo表情位置
x_offset = 20
y_offset = 20

def processFrame(frame):
    # 放大图像
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # 转灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    faceRects = face_cas.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20),maxSize=(200,200))
    if len(faceRects) == 0:  # 如果获取失败
        # print('没能检测出人脸')
        return  # 结束本次循环

    for faceRect in faceRects:
        x, y, w, h = faceRect
        # 框出人脸
        img_addROI = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # 提取人脸部分的像素值
        face_img = gray[y:y + h, x:x + w]
        # print("face:",face_img.shape)
        face_img=cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_LINEAR)
        # 将调整后的图像转为张量
        resized_img = face_img.astype(np.float32)
        tensor_img = torch.from_numpy(resized_img).unsqueeze(0).unsqueeze(0)
        # print(tensor_img.shape)
        # idx,result=predict(tensor_img)
        idx=predict(tensor_img)

        # img_addROI[y_offset:y_offset+emotion_dict[idx].shape[0], x_offset:x_offset+emotion_dict[idx].shape[1]] = cv2.bitwise_and(emotion_dict[idx], emotion_dict[idx], mask=cv2.threshold(emotion_dict[idx], 1, 255, cv2.THRESH_BINARY)[1])
        # 将表情图像覆盖在原始图像的左上角
        img_addROI[y_offset:y_offset + emotion_dict[idx].shape[0], x_offset:x_offset + emotion_dict[idx].shape[1]] = emotion_dict[idx][:, :, :3]
        cv2.imshow("video",img_addROI)

def predict(img):
    prediction = net(img)
    # print(prediction)
    y_pred = np.array([i.detach().numpy() for i in prediction])
    y_pred_labels = np.argmax(y_pred, axis=1)
    npresult = label_binarize(y_pred_labels, classes=np.arange(y_pred.shape[1]))[0]
    # print(npresult)
    idx=np.where(npresult==1)[0][0]

    return idx

cap = cv2.VideoCapture("./video.mp4") # 从视频文件捕获视频
# cap = cv2.VideoCapture(0)
while cap.isOpened():# 读视频
    ret, frame = cap.read()
    if ret:
        #在窗口中显示视频
        # cv2.imshow('video', frame)
        # print(frame.shape)
        processFrame(frame)

# 读键盘
        key = cv2.waitKey(40)
        if key & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()






