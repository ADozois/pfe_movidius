import tensorflow as tf
import argparse
import os
import cv2
import numpy as np
import csv
import datetime

def write_data_to_file(file, start_list, end_list):
    fieldname = ['delta']
    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldname)

        for i in range(0, len(start_list)):
            delta = end_list[i] - start_list[i]
            writer.writerow({'delta': str(delta)})

if __name__ == '__main__':

    graph_path = "model/ssd_mobilenet/frozen_inference_graph.pb"
    treshold = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', required=True, type=int)
    args = parser.parse_args()


    nbr_iter = args.iter

    csv_path = "data_delta_tf_" + str(nbr_iter) + ".csv"
        
    print("Loading model")

    load_start = datetime.datetime.now()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    load_end = datetime.datetime.now()
    
    load_delta = load_end - load_start

    print("Model loaded in: " + str(load_delta) + " s")

    cam = cv2.VideoCapture(0)

    count = 0
    start = []
    end = []
    print("Starting detection")
    loop_start = datetime.datetime.now()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            
            while count < nbr_iter:
                _, img = cam.read()

                image_np_expanded = np.expand_dims(img, axis=0)

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                start.append(datetime.datetime.now())
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                end.append(datetime.datetime.now())

                count += 1

    print("Detection finish")

    loop_end = datetime.datetime.now()
    delta_loop = loop_end - loop_start

    total = loop_end - load_start

    write_data_to_file(csv_path, start, end)

    print("Detection time: " + str(delta_loop) + " s")

    print("Total time: " + str(total) + " s")        


    


    