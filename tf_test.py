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


def load_image_into_numpy_array(image):
  im_width, im_height = image.shape[:2]
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


if __name__ == '__main__':

    graph = "frozen_inference_graph.pb"
    treshold = 0.5

    csv_path = "data_delta_tf.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()
    dir_path = args.path
        

    graph_path = os.path.join(dir_path, graph)

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
    while count < 100:
        _, frame = cam.read()

        #img = load_image_into_numpy_array(frame)
        img = frame
        start.append(datetime.datetime.now())
        output = run_inference_for_single_image(img, detection_graph)
        end.append(datetime.datetime.now())

        # for i in range(0, output['num_detections']):
        #     if output['detection_scores'] > treshold:
        #         print(output['detection_classes'])
        count += 1
    
    print("Detection finish")

    loop_end = datetime.datetime.now()
    delta_loop = load_end - loop_start

    total = loop_end - load_start

    write_data_to_file(csv_path, start, end)

    print("Detection time: " + str(delta_loop) + " s")

    print("Total time: " + str(total) + " s")        

    


    