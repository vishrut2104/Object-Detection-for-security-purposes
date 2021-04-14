import cv2
import numpy as np
from moviepy.editor import *

# confidence threshold to detect objects
threshold = 0.55
flag = True #checks if prev frame had overlap
flag1=False
count = 0
bagcreated = False
humancreated = False
list21=[]
list22 = []

class item:
    def __init__(self, name, x, y, w, h, ifOverlap=False):
        self.name = name
        self.left = x
        self.top = y
        self.width = w
        self.height = h
        self.right = self.left + self.width
        self.bottom = self.top + self.height


def ifoverlap(item1, item2):
    l1 = [item1.left, item1.top]
    l2 = [item2.left, item2.top]
    r2 = [item2.right, item2.bottom]
    r1 = [item1.right, item1.bottom]

    # If one rectangle is on left side of other
    if l1[0] >= r2[0] or l2[0] >= r1[0]:
        return False

    # If one rectangle is above other
    if r1[1] <= l2[1] or r2[1] <= l1[1]:
        return False

    return True


# to make a video using webcam we need to make an object of the VideoCapture function of openCV.


# cap = cv2.VideoCapture(0)  # cap is an object of inbuilt videocapture class, (0) means webcam.

cap = cv2.VideoCapture("videoplayback.mp4")

# cap.set is used to set different properties of the video window such as width(3) and height(4)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

# made a list of names from coco.names dataset
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# settings for the window and detection
# same as documentation

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# tracker = cv2.TrackerCSRT_create()
def drawBox(img,left,top,width,height):
    cv2.rectangle(img, (left, top), (left + width, top + height), color=(255, 255, 255), thickness=3)

#video loop
while True:
    tc = cv2.getTickCount()
    success, img = cap.read()

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-tc)
    cv2.putText(img,"FPS : "+str(int(fps)),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    classIds, confs, bbox = net.detect(img, confThreshold=threshold)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))


    # applying non maximum supression to remove false detection
    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold=0.4)
    # print(indices)
    for ind in indices:
        # print(ind)
        ind = ind[0]
        box = bbox[ind]
        left, top, width, height = box[0], box[1], box[2], box[3]



        #left = left most line of the box
        #top = top most line of box
        #right = right most line of box
        #bottom = bottom most line of box

        if(classIds[ind][0]==1 or classIds[ind][0]==25 or classIds[ind][0]==27 or classIds[ind][0]==33 ):
            if(classIds[ind][0]==1):
                ob1 = item('person',left,top,width,height)
                # print("person detected")
                humancreated = True
                list21.append(ob1)
                # print(len(list21))
                # if success:
                drawBox(img, left, top, width, height)


            if(classIds[ind][0] ==33 or classIds[ind][0] ==27 or classIds[ind][0]==25):
                ob2 = item('SUITCASE',left,top,width,height)
                bagcreated = True
                # print("BAGCREATED")
                list22.append(ob2)
                # if success:
                # drawBox(img, left, top, width, height)

            # success, bbox = tracker.update(img)


            # print("***********")
            # print(f"{left},{top},{width},{height}")
            # print(classNames[classIds[ind][0]-1])
            # print("***********")
            cv2.rectangle(img, (left, top), (left + width, top + height), color=(0, 0, 255), thickness=3)
            cv2.putText(img, classNames[classIds[ind][0] - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_ITALIC, 1,
                            color=(0, 255, 0), thickness=2)
            # success,bbox = tracker.update(img)






    #for loop ends
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

    if bagcreated and humancreated:
        # tracker.init(img,bbox)
        for bag in list22:
            for person in list21:
                overlap2 = ifoverlap(bag,person)
                if (overlap2):
                    flag1 = True
                    break
                else:
                    flag1 = False

        if (flag1):
            flag = True
        else:
            flag = False

        if (flag):
            count = 0
            print(" not abandoned")
        else:
            count = count + 1
            print(f"bag abandoned : {count} ")

        if (count >= 500):
            print("ABANDONED OBJECT DETECTED")
            cap.release()
            cv2.destroyAllWindows()
            exit()
    else:
        continue

    list21.clear()
    list22.clear()