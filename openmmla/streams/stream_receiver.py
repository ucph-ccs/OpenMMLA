class StreamReceiver:
    def __init__(self, source, **kwargs):
        self.source = source    # e.g. 'pyaudio', 'socket', 'cv2'
        self.config = kwargs
        self.stream = None

    def start(self):
        raise NotImplementedError("Start method must be implemented by subclasses.")

    def stop(self):
        raise NotImplementedError("Stop method must be implemented by subclasses.")

    def read(self):
        raise NotImplementedError("Read method must be implemented by subclasses.")

    def write(self, data):
        raise NotImplementedError("Write method must be implemented by subclasses.")
