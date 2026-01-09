import socket
import json
import time
from typing import Optional, Dict, Any, Tuple
import threading
import queue


class Telemetry:
    """Helper class to send and receive telemetry data from Assetto Corsa via UDP sockets.

    - Sends commands/actions to AC on port 9877
    - Receives telemetry data from AC on port 9876
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        send_port: int = 9877,
        recv_port: int = 9876,
        timeout: float = 0.1,
        auto_start_receiver: bool = False,
    ):
        self.host = host
        self.send_port = send_port
        self.recv_port = recv_port
        self.timeout = timeout

        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind((self.host, self.recv_port))
        self.recv_socket.settimeout(self.timeout)

        self._receiver_thread = None
        self._receiver_running = False
        self._latest_data = None
        self._data_queue = queue.Queue(maxsize=100)
        self._lock = threading.Lock()

        if auto_start_receiver:
            self.start_receiver()

    def send_command(self, data: Dict[str, Any]) -> None:
        """
        Args:
            data: Dictionary to send as JSON
        """
        try:
            message = json.dumps(data).encode("utf-8")
            self.send_socket.sendto(message, (self.host, self.send_port))
        except Exception as e:
            print(f"Error sending command: {e}")

    def send_reset(self) -> None:
        self.send_command({"reset": True})

    def receive_once(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if timeout is not None:
            old_timeout = self.recv_socket.gettimeout()
            self.recv_socket.settimeout(timeout)

        try:
            data, addr = self.recv_socket.recvfrom(65536)

            try:
                text = data.decode("utf-8")
                obj = json.loads(text)
                return obj
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"raw_data": data, "addr": addr}
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
        finally:
            if timeout is not None:
                self.recv_socket.settimeout(old_timeout)

    def start_receiver(self) -> None:
        if self._receiver_running:
            return

        self._receiver_running = True
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True
        )
        self._receiver_thread.start()

    def stop_receiver(self) -> None:
        self._receiver_running = False
        if self._receiver_thread:
            self._receiver_thread.join(timeout=2.0)
            self._receiver_thread = None

    def _receiver_loop(self) -> None:
        """Background loop that continuously receives telemetry."""
        while self._receiver_running:
            data = self.receive_once()
            if data is not None:
                with self._lock:
                    self._latest_data = data

                try:
                    self._data_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self._data_queue.get_nowait()
                        self._data_queue.put_nowait(data)
                    except queue.Empty:
                        pass

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_data

    def get_from_queue(
        self, block: bool = False, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Get telemetry data from queue.

        Args:
            block: Whether to block until data is available
            timeout: Timeout when blocking (None = wait forever)

        Returns:
            Data dict from queue, or None if empty/timeout
        """
        try:
            return self._data_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Clear all pending data from the queue."""
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """Close sockets and stop receiver thread."""
        self.stop_receiver()
        try:
            self.send_socket.close()
        except:
            pass
        try:
            self.recv_socket.close()
        except:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def send_reset(host: str = "127.0.0.1", port: int = 9877) -> None:
    """Quick function to send reset command."""
    with Telemetry(host=host, send_port=port) as telem:
        telem.send_reset()
