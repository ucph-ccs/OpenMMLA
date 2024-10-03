import socket


def clear_socket_udp(sock):
    sock.setblocking(False)
    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
    sock.setblocking(True)


def read_frames_udp(sock, duration, timeout=3):
    sample_rate = 16000  # samples per second
    sample_width = 2  # bytes per sample
    expected_bytes = sample_rate * sample_width * duration
    byte_samples = bytearray()
    sock.settimeout(timeout)
    while len(byte_samples) < expected_bytes:
        try:
            data, addr = sock.recvfrom(2000)
            byte_samples.extend(data)
        except socket.timeout:
            sock.close()
            raise OSError(
                f"No data received for {timeout} seconds, maybe reset your port or check your network connection.")
    return byte_samples


def read_frames_tcp(sock, conn, duration, timeout=3):
    sample_rate = 16000  # samples per second
    sample_width = 2  # bytes per sample
    expected_bytes = sample_rate * sample_width * duration
    byte_samples = bytearray()
    conn.settimeout(timeout)
    while len(byte_samples) < expected_bytes:
        try:
            data = conn.recv(2000)
            byte_samples.extend(data)
        except socket.timeout:
            sock.close()
            conn.close()
            raise OSError(
                f"No data received for {timeout} seconds, maybe reset your port or check your network connection.")
    return byte_samples
