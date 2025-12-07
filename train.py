###   此文件用于训练模型训练的时候有警告但是不影响  #####
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ultralytics-8.2.79-RGBT_2025-03-12\ultralytics\cfg\models\OUR\yolov8-RGBT-Final.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'D:\ultralytics-8.2.79-RGBT_2025-03-12\dataset\FLIR.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='Debug',
                name='test',
                )
