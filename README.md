# **Human following system on Raspberry Pi 4**
---
This system can track human movements and follow a desired person with a Pi camera. It uses a computer vision algorithm like DeepSORT and deploys on embedded hardware such as Raspberry Pi. The instructions for my system are described below:
## How to work
---
This system is built from a pipeline below:
- First, I build a YOLOv4 Tiny model by using an instruction name "darknet" that is used to build a Tiny model. This model is trained on Google Colab with GPU. 

## Reference
- [https://github.com/theAIGuysCode/yolov4-deepsort]
- [https://www.cytrontech.vn/tutorial/face-recognition-using-opencv-on-raspberry-pi-400]
