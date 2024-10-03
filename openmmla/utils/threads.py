import logging
import threading


class RaisingThread(threading.Thread):
    def run(self):
        self._exc = None
        try:
            super().run()
        except Exception as e:
            self._exc = e
            logging.error(f"Error in thread {self.name}: {e}", exc_info=True)

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self._exc:
            print(f"Raising exception {self._exc} from thread {self.name}")
            raise self._exc
