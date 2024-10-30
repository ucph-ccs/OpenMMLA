import sensor
import time
import network
import usocket
import ntptime
import ustruct

SSID = "YOUR_SSID"
KEY = "YOUR_PASSWORD"
VIDEO_BASE_HOSTNAME = "YOUR_VIDEO_BASE_HOSTNAME"
VIDEO_BASE_PORT = 50000

wlan = network.WLAN(network.STA_IF)
wlan.active(True)


def connect_wifi():
    while not wlan.isconnected():
        print("connecting to network...")
        wlan.connect(SSID, KEY)
        time.sleep(3)
    print("network config:", wlan.ifconfig())


connect_wifi()
print("Wi-Fi Connected. Device IP:", wlan.ifconfig()[0])

# Sync time with NTP server
try:
    ntptime.settime()  # Sync RTC with NTP
    print("Time synchronized via NTP:", time.localtime())
except Exception as e:
    print("Failed to sync time:", e)

# Initialize the camera sensor
sensor.reset()
sensor.set_framesize(sensor.QVGA)
sensor.set_pixformat(sensor.RGB565)
sensor.skip_frames(time=2000)


def connect_and_stream():
    while True:
        try:
            client = usocket.socket(usocket.AF_INET, usocket.SOCK_STREAM)
            client.connect(usocket.getaddrinfo(VIDEO_BASE_HOSTNAME, VIDEO_BASE_PORT)[0][-1])
            print("Connected to server at {}:{}".format(VIDEO_BASE_HOSTNAME, VIDEO_BASE_PORT))
            break
        except OSError as e:
            print("Connection error:", e)
            print("Retrying in 5 seconds...")
            time.sleep(5)

    clock = time.clock()

    try:
        while True:
            clock.tick()
            frame = sensor.snapshot()
            cframe = frame.compress(quality=35)

            # Get the current local time
            current_time = time.localtime()  # Get local time tuple
            print("Current time:", current_time)  # Print for debugging

            # Pack the time (year, month, day, hour, minute, second)
            time_bytes = ustruct.pack('>6H', current_time[0], current_time[1], current_time[2],
                                      current_time[3], current_time[4], current_time[5])

            # Send the time first
            client.sendall(time_bytes)

            # Send the size of the compressed frame
            size = cframe.size()
            size_bytes = size.to_bytes(4, 'big')
            client.sendall(size_bytes)

            # Send the compressed frame data
            client.sendall(cframe)
            print("Sent frame of size:", size, "bytes at time:", current_time)
    except OSError as e:
        print("Stream error:", e)
    finally:
        client.close()
        print("Connection closed. Reconnecting...")
        time.sleep(5)
        connect_and_stream()


connect_and_stream()
