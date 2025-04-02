import time

import network
import usocket
from ulab import numpy as np

import audio


SSID = "YOUR_SSID"
PASSWORD = "YOUR_PASSWORD"
AUDIO_BASE_HOSTNAME = "YOUR_AUDIO_BASE_HOSTNAME"
AUDIO_BASE_PORT = 50000  # Change this to the port number you are connecting to

channels = 1
frequency = 16000

wlan = network.WLAN(network.STA_IF)
wlan.active(True)

client = None
addr = None


def connect_wifi():
    while not wlan.isconnected():
        print("connecting to network...")
        wlan.connect(SSID, PASSWORD)
        time.sleep(3)
    print("network config:", wlan.ifconfig())


def connect_to_base():
    global client
    global addr

    # Close any existing socket
    if client:
        client.close()
    client = None
    addr = None

    connect_wifi()

    while True:
        try:
            if wlan.isconnected():
                client = usocket.socket(usocket.AF_INET, usocket.SOCK_DGRAM)
                addr = (usocket.getaddrinfo(AUDIO_BASE_HOSTNAME, AUDIO_BASE_PORT)[0][-1])
                print("Socket created successfully.")
                return
            else:
                connect_wifi()
        except OSError as e:
            if client:
                client.close()
            client = None
            addr = None
            print("Socket creation error:", e)
            time.sleep(3)


def audio_callback(buf):
    global raw_buf
    if raw_buf == None:
        raw_buf = buf


def main():
    global raw_buf

    connect_to_base()

    raw_buf = None
    audio.init(channels=channels, frequency=frequency, highpass=0.9883)
    audio.start_streaming(audio_callback)

    while True:
        try:
            if wlan.isconnected():
                if raw_buf:
                    pcm_buf = np.frombuffer(raw_buf, dtype=np.int16)
                    raw_buf = None
                    n = client.sendto(pcm_buf.tobytes(), addr)
            else:
                print("Disconnected from network.")
                connect_to_base()
        except OSError as e:
            print("OS Error:", e)
            connect_to_base()

    audio.stop_streaming()


if __name__ == '__main__':
    main()
