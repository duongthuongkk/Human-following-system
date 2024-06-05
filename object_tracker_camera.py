import os
import threading
import math
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import face_rec as face
import gesture_rec as gesture
import motor
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video','/home/pi/Desktop/5506755154084.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    tolerance = 0.2 # error of straight direction
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    face_bbox = []
    track_bbox = []
    id_array = []
    face_list = []
    direction_list = []
    host_bbox=[]
    gesture_point=[]
    host_flag = False
    host_id = None
    id_loss = True
    gesture_status=""
    d= None
    # Set default for tracking mode first
    ok_flag = 0
    stop_flag = 0
    pre_ok = 1
    pre_stop = 0
    crop_frame = None
    #checking b-box inside another function
    def check_bbox_inside(bb1:list,bb2:list):
        if not bb1 or not bb2: # Check the list is not empty
            return None
        else:    
            # bounding boxes have form as (xmin,ymin,xmax,ymax)
            if bb1[0]>=bb2[0] and bb1[1]>=bb2[1] and bb1[2]<=bb2[2] and bb1[3]<=bb2[3]:
                return True
            else:
                print("The face bbox isn't insdie track bbox")
                return False
    def return_status(gesture_status,pre_ok,pre_stop):
        ok_flag = pre_ok
        stop_flag = pre_stop
        if gesture_status == "ok":
            ok_flag = 1
            stop_flag = 0
        elif gesture_status == "palm":
            ok_flag = 0
            stop_flag = 1
        else: pass
        return ok_flag,stop_flag

    
    def draw_status(id_loss,ID):
        cv2.rectangle(frame,(0,0),(640,30), (170,255,230), -1)
        cv2.putText(frame, 'HOST ID:',
                  (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                  0.75, (64,64,64), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STATUS:',
                  (210, 20), cv2.FONT_HERSHEY_DUPLEX,
                  0.75, (64,64,64), 2, cv2.LINE_AA)
        cv2.putText(frame, ID,
                  (130, 20), cv2.FONT_HERSHEY_DUPLEX,
                  0.75, (64,64,64), 2, cv2.LINE_AA)
        cv2.putText(frame, 'FPS:',
                  (440, 20), cv2.FONT_HERSHEY_DUPLEX,
                  0.75, (64,64,64), 2, cv2.LINE_AA)
        if ok_flag==1 and id_loss == False:
            cv2.putText(frame, 'FOLLOW',
                  (330, 20), cv2.FONT_HERSHEY_DUPLEX,
                  0.75, (0,255,0), 2, cv2.LINE_AA)
        elif stop_flag==1 and ok_flag == 0 or id_loss == True:
            cv2.putText(frame, 'STOP',
                  (330, 20), cv2.FONT_HERSHEY_DUPLEX,
                  0.75, (255,0,0), 2, cv2.LINE_AA)
    def get_delay(x_deviation):
        deviation=abs(x_deviation)
        if(deviation>=0.4):
            d=0.080
        elif(deviation>=0.35 and deviation<0.40):
            d=0.060
        elif(deviation>=0.20 and deviation<0.35):
            d=0.050
        else:
            d=0.040
        return d
    def move_robot(x_deviation,y,tolerance,d):
        print("x_deviation = ",x_deviation)
        print("y = ",y)
        if abs(x_deviation)<tolerance:
            if y<0.1:
                motor.stop()
                print("Stop following host")
            else:
                motor.go_straight(70)
                print("Go straight")
        else:
            if x_deviation<=-1*tolerance:
                motor.turn_left()
                time.sleep(d)
                print("Turn left")
                motor.stop()
            elif x_deviation>=tolerance:
                motor.turn_right()
                time.sleep(0.02)
                print("Turn right")
                motor.stop()
    #initialize face recogniton
    currentname,data,detector=face.face_int()
    
    #initialize gesture recgonition
    recognizer = gesture.hand_init()
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        ID = 'NONE'
        draw_status(id_loss,ID)
        frame_num +=1
        if frame_num%1==0:
            image_data = cv2.resize(frame, (416,416))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()
            
            # Run gesture prediction
            if crop_frame is not None:
                crop_frame = crop_frame.astype('uint8')
                if stop_flag:
                    image, gesture_result = gesture.get_gesture(frame,recognizer)
                elif stop_flag ==0:
                    image, gesture_result = gesture.get_gesture(crop_frame,recognizer)
            else:
                image, gesture_result = gesture.get_gesture(frame,recognizer)
            frame, gesture_status = gesture.draw_landmarks_on_image(frame,gesture_point,image.numpy_view(),gesture_result)
            ok_flag, stop_flag = return_status(gesture_status,pre_ok,pre_stop)

            if ok_flag:
                # run detections on tflite if flag is set
                if FLAGS.framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    # run detections using yolov3 if flag is set
                    if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                else:
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=FLAGS.iou,
                    score_threshold=FLAGS.score
                )

                # convert data to numpy arrays and slice out unused elements
                num_objects = valid_detections.numpy()[0]
                bboxes = boxes.numpy()[0]
                bboxes = bboxes[0:int(num_objects)]
                scores = scores.numpy()[0]
                scores = scores[0:int(num_objects)]
                classes = classes.numpy()[0]
                classes = classes[0:int(num_objects)]

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(bboxes, original_h, original_w)

                # store all predictions in one parameter for simplicity when calling functions
                pred_bbox = [bboxes, scores, classes, num_objects]

                # read in all class names from config
                class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                # by default allow all classes in .names file
                allowed_classes = list(class_names.values())
                
                # custom allowed classes (uncomment line below to customize tracker for only people)
                #allowed_classes = ['person']

                # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                names = []
                deleted_indx = []
                for i in range(num_objects):
                    class_indx = int(classes[i])
                    class_name = class_names[class_indx]
                    if class_name not in allowed_classes:
                        deleted_indx.append(i)
                    else:
                        names.append(class_name)
                names = np.array(names)

                # delete detections that are not in allowed_classes
                bboxes = np.delete(bboxes, deleted_indx, axis=0)
                scores = np.delete(scores, deleted_indx, axis=0)

                # encode yolo detections and feed to tracker
                features = encoder(frame, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                #initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]       


                # Call the tracker
                tracker.predict()
                tracker.update(detections)
                frame_size = np.shape(frame)
                # Turn on face recognition to find host when host flag isn't set
                if host_flag == False:
                    frame, face_bbox = face.get_frame(frame,currentname,data,detector)
                        
                # update tracks
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    track_bbox.append(int(bbox[0])) #xmin
                    track_bbox.append(int(bbox[1])) #ymin
                    track_bbox.append(int(bbox[2])) #xmax
                    track_bbox.append(int(bbox[3])) #ymax
                    id_array.append(track.track_id)
                    # Turn on flag when determine the host
                    if check_bbox_inside(face_bbox,track_bbox):
                        host_flag = True
                        host_id = track.track_id
                    if track.track_id == host_id:
                        color = (255,0,0)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len("host")+len(str(track.track_id))+len("-ID:"))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, "host" + " -ID:" + str(host_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                        
                        host_bbox.append(round(track_bbox[0]/frame_size[1],3))   #xmin_normalize
                        host_bbox.append(round(track_bbox[1]/frame_size[0],3))   #ymin_normalize
                        host_bbox.append(round(track_bbox[2]/frame_size[1],3))   #xmax_normalize
                        host_bbox.append(round(track_bbox[3]/frame_size[0],3))   #ymax_normalize
                        gesture_point = [int(bbox[0]), int(bbox[1])]
                        # Create central point
                        xmean_bbox =  (bbox[2]-bbox[0])/2
                        ymean_bbox =  (bbox[3]-bbox[1])/2
                        central_point = (int(bbox[0] + xmean_bbox),int(bbox[1] + ymean_bbox))
                        frame = cv2.circle(frame, central_point, 5, (255,255,0), 5)
                        normal_point = (round((bbox[0] + xmean_bbox)/frame_size[1],3),round((bbox[0] + ymean_bbox)/frame_size[0],3))
                        # Calculate the distance from camera to central point
                        central_bottom = (int(frame_size[1]/2),int(frame_size[0]))


                        
                        # Crop image to concentrate host's hand
                        if all(num>0 for num in track_bbox) and track_bbox[1]<track_bbox[3] and track_bbox[0]<track_bbox[2]:
                            crop_frame = frame[track_bbox[1]:track_bbox[3],track_bbox[0]:track_bbox[2]]

                    # draw bbox on screen
                    else:
                        color = (0,0,255)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id))+len("-ID:"))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name + " -ID:" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                     # Delete the track box and id value for each iteration
                    track_bbox.clear()
                # Turn off host flag if don't find the host id
                if host_id not in id_array: 
                    host_flag = False
                    id_loss = True
                ID = str(host_id)

                #Control bot track to person
                normalize_frame = frame.astype(np.float32)
                normalize_frame = cv2.normalize(frame,None,0,1.0, cv2.NORM_MINMAX)
                normalize_frame = np.round(normalize_frame,3)
                if host_bbox:
                    id_loss = False
                    x_deviation = round(normal_point[0] - 0.5,3)
                    print("y_max =", host_bbox[3])
                    y = 1 - host_bbox[3]
                    d = get_delay(x_deviation)
                    thread = threading.Thread(target = move_robot,args = (x_deviation,y,tolerance,d))
                    thread.start()
                    #move_robot(x_deviation,y,tolerance,d)
                draw_status(id_loss,ID)
                id_array.clear()
                face_bbox.clear()

            elif stop_flag:
                pass    
            
            pre_ok = ok_flag
            pre_stop = stop_flag
            face_list.clear()
            direction_list.clear()
            host_bbox.clear()
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame,str(round(fps,2)),(510,20),cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0),2,cv2.LINE_AA)
            motor.stop()
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not FLAGS.dont_show:
                cv2.imshow("Result",result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
   
    vid.release()
    motor.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
