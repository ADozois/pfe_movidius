import tensorflow as tf
import argparse
import os

def main():
    pass

if __name__ == '__main__':

    graph = "frozen_inference_graph.pb"
    label = "label_map.pbtxt"

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    
    args = vars(parser.parse_args())

    dir_path = args["path"]

    graph_path = os.path.join(dir_path, graph)
    label_path = os.path.join(dir_path, label)

    


    