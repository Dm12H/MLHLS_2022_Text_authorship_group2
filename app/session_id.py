from threading import Lock

class SessionId:
    def __init__(self):
        self.id = 0
        self._lock = Lock()

    def get_session_id(self):
        with self._lock:
            new_session = self.id
            self.id += 1

        return new_session