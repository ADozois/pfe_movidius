

class NCSQueue:
    def __init__(self, queue=None):
        self._queue = queue
        self._count = 0

    def add_elem(self, image_tensor, object=None):
        self._queue.write_elem(image_tensor, object)
        self._count += 1

    @property
    def count(self):
        return self._count

    def empty(self):
        if self._count == 0:
            return True
        else:
            return False

    def inference(self):
        if self._count > 0:
            self._count -= 1

    def destroy(self):
        self._count = 0
        if self._queue is not None:
            self._queue.destroy()
