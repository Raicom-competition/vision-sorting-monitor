import socket
import threading
import time


class RobotOnlineClient:
    def __init__(self, host="192.168.159.1", port=2001, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = None
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._socket is not None

    def connect(self):
        if self._socket is not None:
            return
        self._socket = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self._socket.settimeout(self.timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def disconnect(self):
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        self._socket = None

    @staticmethod
    def _has_six_numbers(data):
        text = data.decode("ascii", errors="ignore").replace(" ", "")
        parts = [part for part in text.split(",") if part]
        return len(parts) >= 6

    def send_and_receive(self, command):
        if self._socket is None:
            raise ConnectionError("机器人未连接")
        with self._lock:
            command_text = command.strip()
            is_getpose = command_text.lower() == "getpose"
            self._socket.sendall(command_text.encode("ascii"))

            if not is_getpose:
                return ""

            buffer = b""
            self._socket.settimeout(0.5)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                try:
                    chunk = self._socket.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk
                if self._has_six_numbers(buffer):
                    break
                if len(buffer) > 65536:
                    break
            return buffer.decode("ascii", errors="ignore").strip()
