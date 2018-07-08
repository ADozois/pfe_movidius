from NCS import NCS
from skimage import io
import numpy as np
import cv2

def preprocess_image(src):

    # scale the image
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
        new_img = normalize(new_img)

    return new_img


def normalize(img, max_value=None):
    img = img - 127.5
    img = img * 0.007843
    print(np.min(img))
    return img

if __name__=="__main__":
    image_path = "/home/walle/Movidius/ncappzoo/data/images/nps_chair.png"
    graph_path = "/home/walle/pfe_movidius/model/test/graph"
    name = "Test"

    cam = cv2.VideoCapture(0)
    #image = io.imread(image_path)
    image = cv2.imread(image_path)
    #image = image[:, :, ::-1]

    test = NCS(name, graph_path)


#    while True:
#    _, frame = cam.read()
#        img = cv2.resize(frame, (300, 300)).astype(np.float16)

    img = preprocess_image(image)

    img = img.astype(np.float16)
    #test.add_image(image)
    test.execute_inference_with_tensor(img)
    test.execute_inference()
    test.execute_inference()

    pred = test.get_prediction()

    label = np.argmax(pred)

    test.close()


