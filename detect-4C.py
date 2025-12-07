###   此为检测文件    ###

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"D:\ultralytics-8.2.79-RGBT_2025-03-12\dronevehicle.pt") # select your model.pt path
    model.predict(source=r"D:\ultralytics-8.2.79-RGBT_2025-03-12\dronevehicle-inference\20251011194140_41_21.jpg",
                  imgsz=512,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBT",
                  channels=4,
                  save=True,
                  # conf=0.2,
                  #visualize=True # visualize model features maps
                )
