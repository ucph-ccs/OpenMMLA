#include <PDM.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// WiFi network credentials
char ssid[] = "YOUR_SSID";
char pass[] = "YOUR_Password";

// UDP server
WiFiUDP udp;

// PDM settings
const char channels = 1;
const int frequency = 16000;
short sampleBuffer[512];
volatile int samplesRead;

// Client IP and port for UDP
IPAddress clientIP(172, 20, 10, 4);  // replace with your client's IP
// IPAddress clientIP(192, 168, 31, 92);
unsigned int clientPort = 50001;

void setup() {
  // Serial.begin(9600);
  // while (!Serial);

  // Initialize PDM
  PDM.onReceive(onPDMdata);
  PDM.setGain(24);
  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  // Initialize WiFi
  while (WiFi.status() != WL_CONNECTED) {
    Serial.println(ssid);
    int status = WiFi.begin(ssid, pass);
    if (status != WL_CONNECTED) {
      Serial.print("Failed to connect to WiFi. Status code: ");
      Serial.println(status);
    }
    Serial.println(WiFi.status());
    delay(10000);
  }

  // Print WiFi status
  printWifiStatus();

  // Start the server
  udp.begin(clientPort);
}

void loop() {
  if (samplesRead) {
    udp.beginPacket(clientIP, clientPort);
    udp.write((uint8_t *)sampleBuffer, samplesRead * sizeof(short));
    udp.endPacket();
    samplesRead = 0;
  }
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
