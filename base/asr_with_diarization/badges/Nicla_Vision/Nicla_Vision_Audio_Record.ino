#include <WiFi.h>
#include <PDM.h>
#include <SPI.h>
#include <WiFiClient.h>

#include "arduino_secrets.h"

#define START_BYTE 0x01
#define STOP_BYTE 0x02
#define HEARTBEAT_BYTE 0x00

char ssid[] = SECRET_SSID;        
char pass[] = SECRET_PASS;        
int keyIndex = 0;           

int status = WL_IDLE_STATUS;

// TCP client
WiFiClient client;

static const char channels = 1;
static const int frequency = 16000;

short sampleBuffer[512];

volatile int samplesRead;

IPAddress serverIP(SECRET_IP1, SECRET_IP2, SECRET_IP3, SECRET_IP4);
unsigned int serverPort = 50002;

unsigned long lastHeartBeatTime = 0;
const unsigned long timeout = 60000;
bool heartbeatReceived = false;
bool isRecording = false;

void setup() {
  // Serial.begin(9600);
  // while (!Serial) {
  //   ;
  // }

  if (WiFi.status() == WL_NO_SHIELD) {
    Serial.println("Communication with WiFi module failed!");
    while (true);
  }

  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(3000);
  }
  Serial.println("Connected to wifi");
  printWifiStatus();
  
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
}

void loop() {
  unsigned long currentTime = millis();

  if (client.connected()) {
    while (client.available()) {
      int incomingByte = client.read();
      if (incomingByte == HEARTBEAT_BYTE) {
        heartbeatReceived = true;
        lastHeartBeatTime = currentTime;
        Serial.print("Receive heartbeat. ");
        Serial.print("lastHeartBeatTime: ");
        Serial.println(lastHeartBeatTime);
      } else if (incomingByte == START_BYTE) {
        isRecording = true;
        Serial.println("Start recording.");
      } else if (incomingByte == STOP_BYTE) {
        isRecording = false;
        client.stop();
        Serial.println("Stop recording.");
        delay(3000);
      }
    }

    if (currentTime - lastHeartBeatTime > timeout) {
      Serial.print("Time out. ");
      Serial.print("Current time: ");
      Serial.print(currentTime);  // Print out the current time
      Serial.print(" lastHeartBeatTime: ");
      Serial.println(lastHeartBeatTime);
      heartbeatReceived = false;
      isRecording = false;
      client.stop();
      delay(3000);
    }

    if (isRecording && samplesRead) {
      client.write((uint8_t *)sampleBuffer, samplesRead * sizeof(short));
      samplesRead = 0;
    }
  }

  else {
    delay(3000);

    if (!client.connect(serverIP, serverPort)) {
      Serial.println("Failed to connect to server");
      client.stop();
    } else {
      Serial.println("Connected to server");
      // Reset the last check time and heartbeat received flag
      lastHeartBeatTime = currentTime;
      heartbeatReceived = true;
    }
  }
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable > sizeof(sampleBuffer)) {
    Serial.println("Oops, buffer exceeds");
    bytesAvailable = sizeof(sampleBuffer);
  }

  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

void printWifiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
