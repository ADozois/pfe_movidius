import tensorflow as tf
import argparse
import os
import cv2
import numpy as np
import csv
import datetime
from utils import label_map_util

MIN_SCORE = 0.5

def draw_boundingbox(image, left, right, up, down, class_name, confidence):
    image_width = image.shape[1]
    image_height = image.shape[0]

    base_index = 0
    class_id = class_name
    percentage = confidence
    if (percentage <= MIN_SCORE):
        return

    label_text = class_name + " " + str(confidence) + " %"
    box_left = int(left * image_width)
    box_top = int(up * image_height)
    box_right = int(right * image_width)
    box_bottom = int(down * image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 3
    cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    scale_max = (100.0 - MIN_SCORE)
    scaled_prob = (percentage - MIN_SCORE)
    scale = scaled_prob / scale_max

    # draw the classification label string just above and to the left of the rectangle
    #label_background_color = (70, 120, 70)  # greyish green background for text
    label_background_color = (0, int(scale * 175), 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return image

def write_data_to_file(file, start_list, end_list):
    fieldname = ['delta']
    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldname)

        for i in range(0, len(start_list)):
            delta = end_list[i] - start_list[i]
            writer.writerow({'delta': str(delta)})

def load_labelmap(path):
        print('Loading labelmap from label_map.pbtxt')
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        return label_map_util.create_category_index(categories)

if __name__ == '__main__':

    graph_path = "model/ssd_mobilenet/frozen_inference_graph.pb"
    label_path = "model/ssd_mobilenet/label_map.pbtxt"
    treshold = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', required=True, type=int)
    args = parser.parse_args()


    nbr_iter = args.iter

    csv_path = "data_delta_tf_" + str(nbr_iter) + ".csv"

    classes_name = load_labelmap(label_path)
        
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
        init_op = tf.global_variables_initializer()
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

                for i in range(0, len(boxes)):
                    box = boxes[0][i]
                    name = classes_name[classes[0][i]]
                    name = name['name']
                    img = draw_boundingbox(img, box[1], box[3], box[0], box[2], name, scores[0][i])

                cv2.imshow("Detection", img)
                cv2.waitKey(1)

    print("Detection finish")

    loop_end = datetime.datetime.now()
    delta_loop = loop_end - loop_start

    total = loop_end - load_start

    write_data_to_file(csv_path, start, end)

    print("Detection time: " + str(delta_loop) + " s")

    print("Total time: " + str(total) + " s")        


    


    