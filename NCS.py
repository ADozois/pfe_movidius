from mvnc import mvncapi as mvnc
from NCSQueue import NCSQueue


class NCS:
    def __init__(self, name, model_path):
        self._device = None
        self._get_device()
        self._graph = mvnc.Graph(name)
        self._input = None
        self._output = None
        self._labels = None
        self.allocate_model(model_path)

    @staticmethod
    def _detect_devices():
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
            raise Exception("No device found")
        return devices

    def _get_device(self):
        devices = self._detect_devices()
        self._device = mvnc.Device(devices[0])
        self.open_device()

    def open_device(self):
        self._device.open()

    def allocate_model(self, graph_path):
        graph_buffer = self._load_graph(graph_path)
        self._input, self._output = self._graph.allocate_with_fifos(
            self._device, graph_buffer, input_fifo_data_type=mvnc.FifoDataType.FP16, output_fifo_data_type=mvnc.FifoDataType.FP16)

    @staticmethod
    def _load_graph(graph_path):
        with open(graph_path, "r") as f:
            graph_bin = f.read()
        return graph_bin

    def add_image(self, image_tensor, usr_obj=None):
        self._input.write_elem(image_tensor, usr_obj)

    def get_prediction(self):
       # if not self._input.empty():
        prediction, _ = self._output.read_elem()
        return prediction
       # else:
       #     return None

    def execute_inference(self):
        self._graph.queue_inference(self._input, self._output)

    def execute_inference_with_tensor(self, tensor, usr_obj=None):
        self._graph.queue_inference_with_fifo_elem(
            self._input, self._output, tensor, usr_obj)

    def close(self):
        self._destroy_queue()
        self._destroy_graph()
        self._device.close()
        self._device.destroy()

    def _destroy_queue(self):
        self._input.destroy()
        self._output.destroy()
        self._input = None
        self._output = None

    def _destroy_graph(self):
        self._model.destroy()
        self._model = None

    def _convert_to_label(self, index):
        return self._labels[index]
