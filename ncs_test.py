from NCS import NCS
import argparse
import os
import cv2
import numpy as np
import csv
import datetime

def preprocess_image(src):

    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img

def write_data_to_file(file, start_list, end_list):
    fieldname = ['delta']
    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldname)

        for i in range(0, len(start_list)):
            delta = end_list[i] - start_list[i]
            writer.writerow({'delta': str(delta)})

if __name__ == '__main__':

    graph_path = "/model/ssd_caffe/graph"
    treshold = 0.5

    csv_path = "data_delta_ncs.csv"

    cam = cv2.VideoCapture(0)

    print("Loading model")

    load_start = datetime.datetime.now()

    detection = NCS("test", graph_path)

    load_end = datetime.datetime.now()
    
    load_delta = load_end - load_start

    print("Model loaded in: " + str(load_delta) + " s")

    count = 0
    start = []
    end = []
    print("Starting detection")
    loop_start = datetime.datetime.now()
    while count < 100:
        _, frame = cam.read()

        img = preprocess_image(frame)

        #img = load_image_into_numpy_array(frame)
        start.append(datetime.datetime.now())
        detection.execute_inference_with_tensor(img.astype(np.float16))
        output = detection.get_prediction()
        end.append(datetime.datetime.now())

        # for i in range(0, output['num_detections']):
        #     if output['detection_scores'] > treshold:
        #         print(output['detection_classes'])
        count += 1

     print("Detection finish")

    loop_end = datetime.datetime.now()
    delta_loop = loop_end - loop_start

    total = loop_end - load_start

    detection.close()

    write_data_to_file(csv_path, start, end)

    print("Detection time: " + str(delta_loop) + " s")

    print("Total time: " + str(total) + " s")        

    


    