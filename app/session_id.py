from threading import Lock

class SessionId:
    def __init__(self):
        self.id = 0
        self._lock = Lock()

    def get_session_id(self):
        new_session = self.id

        with self._lock:
            self.id += 1

        return new_session