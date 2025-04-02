import network
import time
import image
import json
import sensor
from mqtt import MQTTClient
from pyb import LED

SSID = "<YOUR_SSID>"
PASSWORD = "<YOUR_PASSWORD>"
BROKER = "<YOUR_BROKER_HOST_NAME>"
PORT = 1883  # Default MQTT port
TOPIC = "<SELECTED_SESSION_BUCKET>/video"  # MQTT topic to publish messages
SENDER_ID = '0'  # Unique ID for the Badge
TAG_FAMILIES = image.TAG36H11

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
client = None


def connect_wifi():
    while not wlan.isconnected():
        print("connecting to network...")
        wlan.connect(SSID, PASSWORD)
        time.sleep(3)
    print("network config:", wlan.ifconfig())


def connect_to_server():
    global client

    if client and client.sock:
        client.sock.close()
    client = None

    connect_wifi()

    while True:
        try:
            if wlan.isconnected():
                client = MQTTClient(SENDER_ID, server=BROKER, port=PORT, ssl=False, keepalive=60)
                client.connect()
                print("Connect to MQTT server successfully.")
                return
            else:
                connect_wifi()
        except OSError as e:
            if client and client.sock:
                client.sock.close()
            client = None
            print(f"MQTT client creation error: {e}")
            time.sleep(3)


def main():
    """Main function for detecting tags and publishing results."""
    connect_to_server()

    red_led = LED(1)
    green_led = LED(2)
    blue_led = LED(3)

    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QQVGA)
    sensor.set_vflip(True)
    sensor.skip_frames(time=2000)
    clock = time.clock()

    while True:
        try:
            if wlan.isconnected():
                clock.tick()
                img = sensor.snapshot()
                tags = img.find_apriltags(families=TAG_FAMILIES)
                detected_tags = [tag.id() for tag in tags]

                # Update LEDs based on tag detection
                red_led.off()
                green_led.off()
                blue_led.off()

                if len(tags) == 0:
                    red_led.on()
                elif len(tags) == 1:
                    green_led.on()
                elif len(tags) > 1:
                    blue_led.on()

                for tag in tags:
                    img.draw_rectangle(tag.rect(), color=(255, 0, 0))
                    img.draw_cross(tag.cx(), tag.cy(), color=(255, 0, 0))

                message = {"sender_id": SENDER_ID, "detected_tags": detected_tags}
                if detected_tags:
                    print(message)
                    message_str = json.dumps(message)
                    client.publish(TOPIC, message_str, qos=0)
                time.sleep_ms(50)
            else:
                connect_to_server()
        except Exception as e:
            print(f"Error {e}, trying to reconnect to MQTT server.")
            connect_to_server()


if __name__ == "__main__":
    main()
