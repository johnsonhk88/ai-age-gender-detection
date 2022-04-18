import cv2 as cv
import math
import time
import argparse
import os 
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS,cross_origin
import io
from numpy import argmax
from PIL import Image
import base64
import numpy as np

# inital Flask 
app = Flask(__name__)
# install 
socketio = SocketIO(app, cors_allowed_origins='*')

currentDir = os.getcwd()
# print(currentDir)
# currentDir = "/home/johnsonhk88/Full-Stack-Project/simple-videochat-webrtc/python/"

absolutepath = os.path.abspath(__file__)
print(absolutepath)

fileDirectory = os.path.dirname(absolutepath)
print(fileDirectory)

#Path of parent directory
parentDirectory = os.path.dirname(fileDirectory)
print(parentDirectory)

# print("Directory: ", dirname)
#convert string to boolean 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def getFaceBox(net, frame, threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0] # get image height
    frameWidth  = frameOpencvDnn.shape[1] # get image width 
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections= net.forward()  # predict
    bboxes = []  # box list
    for i in range(detections.shape[2]):
        confidence = detections[0,0, i, 2] # get confidence
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3]* frameWidth) # startX
            y1 = int(detections[0, 0, i, 4] * frameHeight) # startY
            x2 = int(detections[0, 0, i, 5] * frameWidth)  # endX
            y2 = int(detections[0, 0, i, 6] * frameHeight) # endY
            bboxes.append([x1, y1, x2, y2])
            # draw boundary box 
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)  
    return frameOpencvDnn , bboxes # retrun image frame and boundary boxes list 


# parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
# parser.add_argument("--input", help="Path to input image or video file. Skip this argument to capture frames from a camera.")
# parser.add_argument("--device", default="cpu", help="Device to inference on")
# parser.add_argument("--threshold", nargs="?", type=float, default=0.70)
# parser.add_argument("--webcamID", nargs="?", type=int, default=0)


# args= parser.parse_args()
#Default 
device = "cpu"
webcamID= -1
threshold = 0.75
usbWebCam = False

#define model and prototype file for face detector, age classifier, gender classifier
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
#model mean values
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#define classification list
ageList = ["(0-2)", "(4-6)",  "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
genderList = ["Male", "Female"]


# Load model network
# print(os.path.join(currentDir, faceModel))
pathfaceModel = os.path.join(currentDir, faceModel)
pathfaceProto = os.path.join(currentDir, faceProto)
pathageModel = os.path.join(currentDir, ageModel)
pathageProto = os.path.join(currentDir, ageProto)
pathgenderModel = os.path.join(currentDir, genderModel)
pathgenderProto =os.path.join(currentDir, genderProto)

# faceNet = cv.dnn.readNet(faceModel, faceProto)
# ageNet = cv.dnn.readNet(ageModel, ageProto)
# genderNet = cv.dnn.readNet(genderModel, genderProto)

faceNet = cv.dnn.readNet(pathfaceModel, pathfaceProto)
ageNet = cv.dnn.readNet(pathageModel, pathageProto)
genderNet = cv.dnn.readNet(pathgenderModel, pathgenderProto)



#set Opencv CPU or GPU for interferece
if device == "cpu":
    faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    print("Using CPU device")
elif device == "gpu":
    faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    
    
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)

    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    print("Using GPU device")

#Open a video file an an image file or a webcam stream
# cap = cv.VideoCapture(args.input if args.input else args.webcamID)
if usbWebCam == True:
    cap = cv.VideoCapture(webcamID)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G')) # fix the FPS webcam too low, change to MJPG format
    cap.set(cv.CAP_PROP_FPS, 30)

padding = 20
ptime = 0  # previsous timing
ctimg = 0  # current timing


@app.route('/', methods=['POST', 'GET'])
def index():
    #return render_template('index.html')
    return Response("Age Gender Server Runing ..")
    # return app.send_static_file('index.html')


def ageGenderDetect(frame):
    padding = 20
    # Read frame
    ctime = time.time()
    # hasFrame, frame = cap.read()
    # if not hasFrame:
    #     cv.waitKey()
    #     print("Not frame")
    #     return None
    
    print('Resolution: ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0]))

    #face detector
    # frameFace, bboxes = getFaceBox(faceNet , frame, args.threshold)
    frameFace, bboxes = getFaceBox(faceNet , frame)

    if not bboxes:
        print("No face Detected, Checking next frame")
        return None

    for bbox in bboxes:
        # print(bbox)
        #get the face boundary box x1, y1, x2, y2 coordinate  -> y height, x width 
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        #create blob format
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        #find gender
        genderNet.setInput(blob)
        genderPredict = genderNet.forward() # 
        gender = genderList[genderPredict[0].argmax()]
        # print("GenderPredict: ", genderPredict)
        print("Gender: {} , conf = {:.3f}".format(gender, genderPredict[0].max()) )

        # find age
        ageNet.setInput(blob)
        agePredict = ageNet.forward()
        age = ageList[agePredict[0].argmax()]
        # print("Age Output : {}".format(agePredict))
        print("Age : {}, conf = {:.3f}".format(age, agePredict[0].max()))

        #create label for show output
        label = "{},{}".format(gender, age)

        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        # cv.imshow("Age Gender Demo", frameFace)
        # retJpg, jpeg = cv.imencode('.jpg', frameFace)



    #fps show
    # fps = 1 / (ctime - ptime)
    # ptime = ctime 
    # fps =round(fps , 1)
    # strFps = device + " FPS: " + str(fps)
    # print(device , " FPS: ", fps)
    # cv.putText(frameFace, strFps, (0, 140), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1, cv.LINE_AA)
    # cv.imshow("Age Gender Demo", frameFace) 
    # cap.release()
    # cv.destroyAllWindows()
    return frameFace


def getVideo(cap):
    padding = 20
    ptime = 0  # previsous timing
    ctimg = 0  # current timing
    while cap.isOpened():
        # Read frame
        ctime = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            print("Not frame")
            break
        
        print('Resolution: ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0]))

        #face detector
        # frameFace, bboxes = getFaceBox(faceNet , frame, args.threshold)
        frameFace, bboxes = getFaceBox(faceNet , frame)

        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            #get the face boundary box x1, y1, x2, y2 coordinate  -> y height, x width 
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            #create blob format
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            #find gender
            genderNet.setInput(blob)
            genderPredict = genderNet.forward() # 
            gender = genderList[genderPredict[0].argmax()]
            # print("GenderPredict: ", genderPredict)
            print("Gender: {} , conf = {:.3f}".format(gender, genderPredict[0].max()) )

            # find age
            ageNet.setInput(blob)
            agePredict = ageNet.forward()
            age = ageList[agePredict[0].argmax()]
            # print("Age Output : {}".format(agePredict))
            print("Age : {}, conf = {:.3f}".format(age, agePredict[0].max()))

            #create label for show output
            label = "{},{}".format(gender, age)

            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            # cv.imshow("Age Gender Demo", frameFace)
            # retJpg, jpeg = cv.imencode('.jpg', frameFace)



        #fps show
        fps = 1 / (ctime - ptime)
        ptime = ctime 
        fps =round(fps , 1)
        strFps = device + " FPS: " + str(fps)
        print(device , " FPS: ", fps)
        cv.putText(frameFace, strFps, (0, 140), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1, cv.LINE_AA)
        # cv.imshow("Age Gender Demo", frameFace)
        retJpg, jpeg = cv.imencode('.jpg', frameFace) # like imshow but convert into jpg format

        frame = jpeg.tobytes() # extract bytes data 
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    print("Camera Close")    
    cap.release()
    # cv.destroyAllWindows()c


def read64(base64String):
    idx = base64String.find('base64,') # find the base64 text index 
    base64Str = base64String[idx+7:] # get raw data start point after base64, 
  

    # save base64 decoded file
    # with open('basedecode.txt', 'w') as f:
    #     f.write(base64Str)

    # create stream io buff
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64Str)) #write converted image data to buffer
    pimg = Image.open(sbuf)
    tempImg = cv.cvtColor(np.array(pimg), cv.COLOR_RGB2BGR)

    # cv.imwrite("image.jpg", tempImg) # save receive image 

    #convert RGB to BGR
    return tempImg

global fps,prev_recv_time,cnt,fps_array, busy
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]
busy = False

# listen receive image event data
@socketio.on('image')
def image(videoImage):
    global fps, prev_recv_time, cnt, fps_array, busy

    if busy == True:
        return
    busy = True
    recv_time =  time.time()
    # print("Image recv time: ", recv_time)
    text  = "FPS: " + str(fps)
    frame = (read64(videoImage)) # convert into image frame from 

    # face detection 
    resFrame = ageGenderDetect(frame)
    if resFrame is not None:
        # image encode to jpeg
        detectTime = recv_time - prev_recv_time
        print("Image Detect Time: ", detectTime , " s")
        fps = 1/(detectTime)
        fps =round(fps , 1)
        strFps = device + " FPS: " + str(fps)
        prev_recv_time = recv_time
        print(device , " FPS: ", fps)
        cv.putText(resFrame, strFps, (0, 140), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1, cv.LINE_AA)
        imgencode = cv.imencode('.jpeg', resFrame, [ cv.IMWRITE_JPEG_QUALITY, 40])[1] # encode image to jpeg format

        #convert into  base64 encode
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'   # add file header 
        stringData = b64_src + stringData

         # emit the frame back
        emit('response_back', stringData)  #send image  string back to frontend 
    busy = False

       
        # fps_array.append(fps)
        # prev_recv_time = recv_time
        # #print(fps_array)
        # cnt+=1
        # if cnt==30:
        #     fps_array=[fps]
        #     cnt=0
        


# for 
@app.route("/video")
def videoFeed():
    global cap
    return Response(getVideo(cap), mimetype="multipart/x-mixed-replace; boundary=frame") # video 



if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=2204, threaded=True)
    socketio.run(app, port=5000 ,debug=True,  async_mode='eventlet')
    # socketio.run(app, port=5000 ,debug=True,  async_mode="threading")