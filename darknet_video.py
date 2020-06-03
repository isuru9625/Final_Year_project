

from ctypes import *
import math
import random
#import mahotas
import os
import cv2
import numpy as np
import time
import darknet
import matplotlib.pyplot as plt

from numpy import *
from scipy.interpolate import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def initialgetCoordinates(a,b,c,d):
    vehicle=[]
    instance = []
    vehicle.append(a)
    vehicle.append(b)
    vehicle.append(c)
    vehicle.append(d)
    instance.append(vehicle)
    coordinates.append(instance)
    #print(coordinates)

def postCoordinates(vehicles):
    
    used_vehicles = []
    used_coordinates = []
    not_used_coordinates=[]
    #print(len(coordinates))
    
    
    for i in range(0,len(vehicles)):
        current_vehicle =  vehicles[i]
        calc = []
        for j in range(0,len(coordinates)):
            
            current_coordinate = coordinates[j][-1]
            cal = (int(current_vehicle[0])- int(current_coordinate[0]))**2 + (int(current_vehicle[1])- int(current_coordinate[1]))**2 + (int(current_vehicle[2])- int(current_coordinate[2]))**2 + (int(current_vehicle[3])- int(current_coordinate[3]))**2
            calc.append(cal)
            #print(calc)
        #print(calc)
        if(len(calc)>0):
            minimum = min(calc)
            #print(minimum)
            inde = calc.index(minimum)
            #print(inde)
            if(minimum<200):
                coordinates[inde].append(current_vehicle)
                used_coordinates.append(inde)
                used_vehicles.append(i)
    for i in range(0,len(coordinates)):
        if(i not in used_coordinates):
            not_used_coordinates.append(coordinates[i])

    for i in range(0,len(not_used_coordinates)):
        if(not_used_coordinates[i] in coordinates ):
            coordinates.remove(not_used_coordinates[i])


    for i in range(0,len(vehicles)):
        if(i not in used_vehicles):
            coordinates.append([vehicles[i]])


    #print(coordinates)



def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

coordinates = []
def cvDrawBoxes(detections, img,count):
    #dirName = 'crop/processed_images%d' %count
    #try:
        # Create target Directory
        #os.mkdir(dirName)
        #print("Directory " , dirName ,  " Created ") 
    #except FileExistsError:
        #print("Directory " , dirName ,  " already exists")
    
    #mahotas.imsave(dirName+"/aaaaa.jpg",img)
    detected_vehicles = []
    #calculating all the coordinates of all the vehicles in one frame
    for i in range(0,len(detections)):
        
        
        detection = detections[i]
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
       
    




    vehicles = []
    for detection in detections:
        print(detection)
        if(str(detection[0])== "b'traffic light'"):
            print(detection[0])
        var = detections.index(detection)
        
        #print(detection)
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        #print(x,y,w,h)

        
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

        print(xmin, ymin, xmax, ymax,count)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        #cropped_image = img[ymin:ymax, xmin:xmax]
        #cropped_image = img.crop((pt1(0),pt1(1),pt2(0),pt2(1)))
        #mahotas.imsave(dirName+"/image%d.jpg"%var,cropped_image)
        
        
        add = {} 
        
        if(15<abs(pt1[0]-pt2[0])<200 and detection[1]>0.7 and str(detection[0])!="b'traffic light'" ):
            if(count==0):
                initialgetCoordinates(pt1[0],pt2[0],pt1[1],pt2[1])
            else:
                vehicle_i =[]
                vehicle_i.append(pt1[0])
                vehicle_i.append(pt2[0])
                vehicle_i.append(pt1[1])
                vehicle_i.append(pt2[1])
                #print(vehicle_i)
                vehicles.append(vehicle_i)

                
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                      detection[0].decode() +
                      " [" + str(round(detection[1] * 100, 2)) + "]",
                      (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      [0, 255, 0], 2)
    if(count>0):
        #print(vehicles)
        postCoordinates(vehicles)
    img = checkWarning(coordinates,img)
    #print(coordinates)
    return img

def checkWarning(coordinates,img):
    for i in range(0,len(coordinates)):
        current = coordinates[i]
        length = len(current)
        if(length<3):
            total_y = 0
            for j in range(0,length):
                total_y = total_y + current[j][-1]
            average_y = total_y/length
            #print(average_y)
            #if(average_y > 330):
                #print("average warning")
                #cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(255,0,0))
                #break
        else:
            y = []
            y1 = [] #this is the x coordinates

            y.append(current[-3][-1])
            y.append(current[-2][-1])
            y.append(current[-1][-1])
            
            v2 = current[-3][-1] - current[-2][-1]
            v1 = current[-2][-1] - current[-1][-1]

            a = v2 -v1

            average_v = (v1+v2)/2
            s = (average_v*30) + (0.5*a*30*30)


            


            num1 = (current[-3][0] + current[-3][1])/2
            num2 = (current[-2][0] + current[-2][1])/2
            num3 = (current[-1][0] + current[-1][1])/2

            y1.append(num1)
            y1.append(num2)
            y1.append(num3)

            x = [1,2,3]
            #print(y)
            y = array(y)
            x = array(x)
            y1 = array(y1)
            model = polyfit(x,y,1)
            predict = poly1d(model)
            predicted_y = predict(60)

            model1 = polyfit(x,y1,1)
            predict1 = poly1d(model1)
            predicted_x = predict1(60)
            

            #if(420<s<800 and 125<predicted_x<275):
                #cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(255,0,0))
                #break


            print(predicted_y)
            print(s)
            print(predicted_x)
            if(375<predicted_y and 125<predicted_x<325):
                print(y)
                print(predicted_y)

                print(y1)
                
                print(predicted_x)
                print("predict warning")
                cv2.putText(img,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(255,0,0))
                break
            

                          
    return img



netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test.mp4")
    
    count = 0
    cap.set(3, 1280)
    cap.set(4, 720)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    out = cv2.VideoWriter(
        "test.avi", cv2.VideoWriter_fourcc(*"XVID"), fps,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        
       

        
        


        image = cvDrawBoxes(detections, frame_resized,count)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #cv2.putText(image,"warning",(10,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.5,thickness=5,color=(255,255,255))
        out.write(image)
        count = count+1
        #print(1/(time.time()-prev_time))
       
        
        cv2.waitKey(3)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
