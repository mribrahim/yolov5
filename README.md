
## Check yolov5 details from
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [Tutorials](https://github.com/ultralytics/yolov5#tutorials)


## Detect Objects (car, truck and human) + Tracking

download pre-trained weights for vehicle and human detection from aerial images
https://github.com/mribrahim/yolov5-tracking/releases/tag/weightsVehicleHuman

Run inference on example dataset

```
python detect.py --source "/home/ibrahim/Desktop/Dataset/UAV VIVID Tracking Evaluation/egtest02/egtest02/" 
                 --weights ../yolo-weights/car-human.pt --conf-thres 0.25 --view-img

```

![](https://github.com/mribrahim/yolov5-tracking/blob/master/car-track.gif)
