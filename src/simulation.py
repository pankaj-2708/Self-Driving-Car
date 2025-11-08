import cv2
import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import supervision as sv
from custom_objects import mean_iou,dice_loss,tpr,fpr,dice_coefficient

def load_object_detection_model():
    model=YOLO("models/yolo11n.pt")
    return model

def load_lane_detection_model():
    model_path=os.path.join("models","lane_detection.h5")
    model=load_model(model_path,custom_objects={
                       "dice_loss": dice_loss,
                       "mean_iou": mean_iou,
                       "dice_coefficient": dice_coefficient,
                       "tpr": tpr,
                       "fpr": fpr
                   })
    return model

def load_steering_model():
    model_path=os.path.join("models","sterring_angle.h5")
    model=load_model(model_path,custom_objects={'mae': 'mean_absolute_error'})
    return model


def load_image():
    all_files=len(sorted(os.listdir(os.path.join("data","raw","07012018","data"))))
    for i in range(all_files):
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
    start_time=time.time()
    cv2.namedWindow("org_video")
    cv2.namedWindow("sterring")
    cv2.namedWindow("object_detection")
    cv2.namedWindow("lane_detection")
    
    object_model=load_object_detection_model()
    sterring_model=load_steering_model()
    lane_model=load_lane_detection_model()
    
    streeing_wheel=cv2.imread(os.path.join("assets","steering_wheel.jpg"))
    cv2.imshow("sterring",streeing_wheel)
    # steering wheel parameters
    h=200
    w=66
    center = (w // 2, h // 2)
    img_gen=load_image()
    while True:
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
        sterring_rad=sterring_model.predict(img_str)[0][0]
        print("Sterring angle in radians ",sterring_rad)
        sterring_deg=np.degrees(sterring_rad)
        print("Sterring angle in degres ",sterring_deg)
        
        M=cv2.getRotationMatrix2D(center, sterring_deg, 1.0)
        new_sterring_wheel=cv2.warpAffine(streeing_wheel, M, (200, 200))
        print(new_sterring_wheel.shape)
        cv2.imshow("sterring",new_sterring_wheel)

        # object detection
        result=object_model.predict(img,conf=0.25,show=False)[0]
        result=sv.Detections.from_ultralytics(result)
        name_map=object_model.model.names
        obj_det_img=interpret_object_detection_results(result,img,name_map)
        cv2.imshow("object_detection",obj_det_img)
        
        # lane detection
        img_lan=cv2.resize(img,(224,224))
        img_lan=img_lan/255.0
        img_lan=np.expand_dims(img_lan,axis=0)
        img_lan=lane_model.predict(img_lan)[0]
        # img_lan=(img_lan > 0.5).astype(np.uint8)
        cv2.imshow("lane_detection",img_lan)
        
        if cv2.waitKey(100) & 0xFF==ord('q'):
            break
        
        print("Time taken to predict one frame is ",time.time()-start_time)
        
if __name__=="__main__":
    simulate_video_processing()