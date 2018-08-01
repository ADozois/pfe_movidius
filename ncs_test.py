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

def overlay_on_image(image, object_info):
    source_image_width = image.shape[1]
    source_image_height = image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= 0.5):
        return

    label_text = labels[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    scale_max = (100.0 - 0.5)
    scaled_prob = (percentage - 0.5)
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
    cv2.rectangle(image, (0, 0), (100, 15), (128, 128, 128), -1)
    cv2.putText(image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

if __name__ == '__main__':

    graph_path = "model/ssd_caffe/graph"
    treshold = 0.5

    labels = ('background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', required=True, type=int)
    args = parser.parse_args()


    nbr_iter = args.iter

    csv_path = "data_delta_ncs_" + str(nbr_iter) + ".csv"

    cam = cv2.VideoCapture(0)

    print("Loading model")

    load_start = datetime.datetime.now()

    detection = NCS("test", graph_path, labels)

    load_end = datetime.datetime.now()
    
    load_delta = load_end - load_start

    print("Model loaded in: " + str(load_delta) + " s")

    count = 0
    start = []
    end = []
    print("Starting detection")
    loop_start = datetime.datetime.now()
    while count < nbr_iter:
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

        num_valid_boxes = int(output[0])

        for box_index in range(num_valid_boxes):
            base_index = 7 + box_index * 7
            if (not np.isfinite(output[base_index]) or
                    not np.isfinite(output[base_index + 1]) or
                    not np.isfinite(output[base_index + 2]) or
                    not np.isfinite(output[base_index + 3]) or
                    not np.isfinite(output[base_index + 4]) or
                    not np.isfinite(output[base_index + 5]) or
                    not np.isfinite(output[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            x1 = max(int(output[base_index + 3] * img.shape[0]), 0)
            y1 = max(int(output[base_index + 4] * img.shape[1]), 0)
            x2 = min(int(output[base_index + 5] * img.shape[0]), img.shape[0] - 1)
            y2 = min((output[base_index + 6] * img.shape[1]), img.shape[1] - 1)

            # overlay boxes and labels on to the image
            overlay_on_image(frame, output[base_index:base_index + 7])
            cv2.imshow("Detection", frame)
            cv2.waitKey(1)

    print("Detection finish")

    loop_end = datetime.datetime.now()
    delta_loop = loop_end - loop_start

    total = loop_end - load_start

    detection.close()

    write_data_to_file(csv_path, start, end)

    print("Detection time: " + str(delta_loop) + " s")

    print("Total time: " + str(total) + " s")        

    


    