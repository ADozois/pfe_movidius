import mvnc.mvncapi as mvnc
from NCSQueue import NCSQueue


class NCS:
    def __init__(self, name, model_path):
        self._device = None
        self._model = None
        self._input = None
        self._output = None
        self._labels = None
        self._get_device()
        self._create_model(name, model_path)

    @staticmethod
    def _detect_devices():
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
            raise Exception("No device found")
        return devices

    def _get_device(self):
        devices = self._detect_devices()
        self._device = mvnc.Device(devices[0])
        self._device.open()

    def _create_model(self, name, graph_path):
        self._model = mvnc.Graph(name)
        graph_to_load = self._load_graph(graph_path)
        input_fifo, output_fifo = self._model.allocate_with_fifos(self._device, graph_to_load,
                                                                  input_fifo_data_type=mvnc.FifoDataType.FP16,
                                                                  output_fifo_data_type=mvnc.FifoDataType.FP16)
        self._input = input_fifo
        self._output = output_fifo

    @staticmethod
    def _load_graph(graph_path):
        with open(graph_path, "rb") as f:
            graph_bin = f.read()
        return graph_bin

    def add_image(self, image_tensor):
        self._input.write_elem(image_tensor, None)

    def get_prediction(self):
        if not self._input.empty():
            prediction, _ = self._output.read_elem()
            return prediction
        else:
            return None

    def execute_inference(self):
        self._model.queue_inference(self._input, self._output)

    def execute_inference_with_tensor(self, tensor, object=None):
        self._model.queue_inference_with_fifo_elem(self._input, self._output, tensor, object)

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
