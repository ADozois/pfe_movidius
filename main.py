from NCS import NCS
from skimage import io
import numpy as np
import cv2

#import mvnc_simple_api as mvnc

from mvnc import mvncapi as mvnc

LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess_image(src):

    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img

def preprocess(img, width, height, normalize_data=False, max_value=255):
    new_img = cv2.resize(img, (width, height)).astype(np.float16)
    if normalize_data:
        new_img = normalize(new_img.asytpe(np.float16))

    return new_img


def normalize(img, max_value=None):
    img = img - 127.5
    img = img * 0.007843
    print(np.min(img))
    return img

def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    min_score_percent = 60

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

if __name__=="__main__":
    image_path = "/home/walle/Movidius/ncappzoo/data/images/nps_chair.png"
    graph_path = "/home/walle/pfe_movidius/model/ssd_caffe/graph"
    name = "Test"

    cam = cv2.VideoCapture(0)


    #detection = NCS(name, graph_path)
    # devices = mvnc.enumerate_devices()
    # device = mvnc.Device(devices[0])
    # device.open()

    # graph = mvnc.Graph(name)

    # with open(graph_path, 'r') as f:
    #     graph_buffer = f.read()

    # fifo_in, fifo_out = graph.allocate_with_fifos(device, graph_buffer, input_fifo_data_type=mvnc.FifoDataType.FP16,
    #                                                                  output_fifo_data_type=mvnc.FifoDataType.FP16)
    #

    detection = NCS(name, graph_path)


    while True:
        _, frame = cam.read()
        img = preprocess_image(frame)

    #    detection.execute_inference_with_tensor(img)

     #   detection.get_prediction()
        # graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img.astype(np.float16), None)

        # output, usrobj = fifo_out.read_elem()

        detection.execute_inference_with_tensor(img.astype(np.float16))

        output = detection.get_prediction()

        num_valid_boxes = int(output[0])
        print('total num boxes: ' + str(num_valid_boxes))

        for box_index in range(num_valid_boxes):
            base_index = 7 + box_index * 7
            if (not np.isfinite(output[base_index]) or
                    not np.isfinite(output[base_index + 1]) or
                    not np.isfinite(output[base_index + 2]) or
                    not np.isfinite(output[base_index + 3]) or
                    not np.isfinite(output[base_index + 4]) or
                    not np.isfinite(output[base_index + 5]) or
                    not np.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            x1 = max(0, int(output[base_index + 3] * frame.shape[0]))
            y1 = max(0, int(output[base_index + 4] * frame.shape[1]))
            x2 = min(frame.shape[0], int(output[base_index + 5] * frame.shape[0]))
            y2 = min(frame.shape[1], int(output[base_index + 6] * frame.shape[1]))

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                                                                                                             'Confidence: ' + str(
                output[base_index + 2] * 100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')

            # overlay boxes and labels on the original image to classify
            overlay_on_image(frame, output[base_index:base_index + 7])

            cv2.imshow("TEST", frame)
            cv2.waitKey(1)
