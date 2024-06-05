# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

def hand_init():
  base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
  options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=vision.RunningMode.IMAGE)
  recognizer = vision.GestureRecognizer.create_from_options(options)
  return recognizer


def draw_landmarks_on_image(frame,gesture_point,rgb_image, detection_result):
  MARGIN = 10  # pixels
  FONT_SIZE = 1
  FONT_THICKNESS = 2
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.gestures
  annotated_image = frame
  gesture_status = ""
  def frame_or_crop(text_x,text_y,gesture_point):
    if gesture_point:
      x = text_x + gesture_point[0]
      y = text_y + gesture_point[1]
    elif not gesture_point:
      text_x = int(min(x_coordinates) * annotated_image.shape[1])
      text_y = int(min(y_coordinates) * annotated_image.shape[0])
      x = text_x
      y = text_y - MARGIN
    return x,y
      #top_gesture = detection_result.gestures[0][0]
  # Loop through the detected hands to visualize.
  i = 0
  crop_height, crop_width, _ = rgb_image.shape
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    
    # Get the top left corner of the detected hand's bounding box.
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * crop_width)
    text_y = int(min(y_coordinates) * crop_height)
    # Draw handedness (left or right hand) on the image.
    gesture_status = handedness[0].category_name
    x,y = frame_or_crop(text_x,text_y,gesture_point)
    if gesture_status == 'ok':
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (x, y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, (0,255,0), FONT_THICKNESS, cv2.LINE_AA)
    elif gesture_status == 'palm':
      cv2.putText(annotated_image, 'stop',
                  (x, y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, (255,0,0), FONT_THICKNESS, cv2.LINE_AA)
  return annotated_image, gesture_status

def get_gesture(frame,recognizer):
  #frame = cv2.flip(frame, 1)
  # Convert the image from BGR to RGB as required by the TFLite model.
  #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
  # STEP 4: Detect hand landmarks from the input image.
  gesture_result = recognizer.recognize(image)
  # STEP 5: Process the classification result. In this case, visualize it.
  return image, gesture_result
