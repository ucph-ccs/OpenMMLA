import time
import network
import usocket
from ulab import numpy as np
import audio
import ustruct
import ntptime

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

packet_counter = 0
last_sync_time = 0
last_ticks_ms = 0  # Use milliseconds instead of microseconds


def connect_wifi():
    while not wlan.isconnected():
        wlan.connect(SSID, PASSWORD)
        time.sleep(3)


def sync_time():
    global last_sync_time, last_ticks_ms
    try:
        ntptime.settime()  # Sync the time using NTP
        last_sync_time = time.mktime(time.localtime())  # Store the synced time in seconds
        last_ticks_ms = time.ticks_ms()  # Store the current ticks in milliseconds
    except Exception as e:
        pass


def get_current_time():
    global last_sync_time, last_ticks_ms

    # Get the current ticks in milliseconds
    current_ticks_ms = time.ticks_ms()
    elapsed_ms = time.ticks_diff(current_ticks_ms, last_ticks_ms)  # Calculate elapsed time in ms

    # If the elapsed_ms is negative, an overflow has occurred
    if elapsed_ms < 0:
        sync_time()
        elapsed_ms = 0  # Reset the elapsed time to 0 after sync

    # Calculate the elapsed time in seconds and remaining milliseconds
    elapsed_seconds = elapsed_ms // 1000
    elapsed_milliseconds = elapsed_ms % 1000

    # Calculate the current time based on the last sync and elapsed time
    current_time = last_sync_time + elapsed_seconds

    # Format the time as a tuple including milliseconds
    time_struct = time.localtime(current_time)
    return time_struct[0], time_struct[1], time_struct[2], time_struct[3], time_struct[4], time_struct[5], elapsed_milliseconds


def connect_to_base():
    global client, addr
    if client:
        client.close()
    client = None
    addr = None
    connect_wifi()
    sync_time()  # Sync time after connecting to WiFi
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
    global raw_buf, packet_counter, last_sync_time
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

                    # Get the current time using the custom function
                    year, month, day, hour, minute, second, milliseconds = get_current_time()

                    # Combine the packet counter, timestamp string, and audio data
                    time_bytes = ustruct.pack('>I7H', packet_counter, year, month, day, hour, minute, second,
                                              milliseconds)

                    # Combine time bytes with audio data
                    packet_data = time_bytes + pcm_buf.tobytes()

                    n = client.sendto(packet_data, addr)
                    packet_counter += 1
            else:
                print("Disconnected from network.")
                connect_to_base()
        except OSError as e:
            print("OS Error:", e)
            connect_to_base()

    audio.stop_streaming()


if __name__ == '__main__':
    main()
