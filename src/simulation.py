import cv2
import os
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import supervision as sv
import onnxruntime as ort
from custom_objects import mean_iou,dice_loss,tpr,fpr,dice_coefficient
print(ort.get_device())


def load_object_detection_model():
    model=YOLO("models/yolo11n.pt").to("cuda")
    return model

def load_lane_detection_model():
    model=model = ort.InferenceSession(
    "models/lane_detection.onnx",
    providers=["CUDAExecutionProvider"]   # GPU
)
    return model

def load_steering_model():
    model = ort.InferenceSession(
    "models/sterring_angle.onnx",
    providers=["CUDAExecutionProvider"]   # GPU
)
    
    return model


def load_image():
    all_files=len(sorted(os.listdir(os.path.join("data","raw","07012018","data"))))
    for i in range(all_files)[10000:]:
        file=f"{i}.jpg"
        print("Loading file ",file)
        img_path=os.path.join("data","raw","07012018","data",file)
        img=cv2.imread(img_path)
        yield img
        
        
def interpret_object_detection_results(result, img,name_map):

    img=np.array(img)
    for i,lab,conf in zip(result.xyxy,result.class_id,result.confidence):
        cv2.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),color=(0,255,0),thickness=2)
        cv2.putText(img,f"{name_map[lab]} {conf:.2f}",(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return img
        
def simulate_video_processing():
    # frame 1
    cv2.namedWindow("org_video")
    cv2.namedWindow("sterring")
    # cv2.namedWindow("actual_sterring")
    cv2.namedWindow("object_detection")
    cv2.namedWindow("lane_detection")
    
    # actual prediction df
    df=pd.read_csv("./data/raw/07012018/data.csv")
    
    object_model=load_object_detection_model()
    sterring_model=load_steering_model()
    lane_model=load_lane_detection_model()
    
    # Get input / output names
    input_name_sterring = sterring_model.get_inputs()[0].name
    output_name_sterring = sterring_model.get_outputs()[0].name
    input_name_lane = lane_model.get_inputs()[0].name
    output_name_lane = lane_model.get_outputs()[0].name
    
    streeing_wheel=cv2.imread(os.path.join("assets","steering_wheel.png"))
    cv2.imshow("sterring",streeing_wheel)
    # steering wheel parameters
    h=200
    w=66
    center = (w // 2, h // 2)
    img_gen=load_image()
    counter=0
    while True:
        counter+=1
        try:
            img=next(img_gen)
        except StopIteration:
            break
        
        start_time=time.time()
        cv2.imshow("org_video",img)
        
        
        # resizing image for sterring model
        img_str=img[-150:, :, :]
        img_str=cv2.resize(img_str,(h,w))
        img_str=img_str/255.0 -0.5
        img_str=np.expand_dims(img_str,axis=0)
        img_str = img_str.astype(np.float32)

        sterring_rad=sterring_model.run([output_name_sterring],{input_name_sterring:img_str})[0][0][0]
        sterring_deg=sterring_rad*180/np.pi
        
        M=cv2.getRotationMatrix2D(center, sterring_deg, 1.0)
        new_sterring_wheel=cv2.warpAffine(streeing_wheel, M, (250, 250))
        cv2.imshow("sterring",new_sterring_wheel)
        
        # x=df[df['file_name']==f"{counter-1}.jpg"]['angle']
        # M=cv2.getRotationMatrix2D(center, float(x), 1.0)
        # new_sterring_wheel=cv2.warpAffine(streeing_wheel, M, (250, 250))
        # cv2.imshow("actual_sterring",new_sterring_wheel)
        
        # print("Actual sterring angle in degrees ",x)
        print("Predicted steering angle in degrees ",sterring_deg)
        print("Time taken to predict steering angle is ",time.time()-start_time)
        
        # object detection
        start_time=time.time()
        result=object_model.predict(img,conf=0.25,show=False)[0]
        result=sv.Detections.from_ultralytics(result)
        name_map=object_model.model.names
        obj_det_img=interpret_object_detection_results(result,img,name_map)
        cv2.imshow("object_detection",obj_det_img)
        print("Time taken to predict object detection is ",time.time()-start_time)
        
        
        # lane detection
        start_time=time.time()
        img_lan=cv2.resize(img,(224,224))
        img_lan=img_lan/255.0
        img_lan=np.expand_dims(img_lan,axis=0)
        img_lan=img_lan.astype(np.float32)
        img_lan=lane_model.run([output_name_lane],{input_name_lane:img_lan})[0][0]
        # img_lan=(img_lan > 0.5).astype(np.uint8)
        cv2.imshow("lane_detection",img_lan)
        print("Time taken to predict lane detection is ",time.time()-start_time)        
        if cv2.waitKey(100) & 0xFF==ord('q'):
            break
        
        
if __name__=="__main__":
    simulate_video_processing()