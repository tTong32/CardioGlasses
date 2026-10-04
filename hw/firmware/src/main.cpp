// CardioGlasses - ESP32 + MAX30102 + MPU6050 -> BLE stream
// Wiring (DevKit V1): both sensors share the I2C bus
//   MAX30102: VIN->3V3, GND->GND, SDA->D21, SCL->D22
//   MPU6050 : VCC->3V3, GND->GND, SDA->D21, SCL->D22 (AD0 unconnected -> 0x68)
//
// Packet = K samples back to back, 24 bytes each (little-endian):
//   uint32 idx, uint32 ir, uint32 red, int16 ax, ay, az, gx, gy, gz
//   Accel +-4 g (8192 LSB/g), gyro +-500 dps (65.5 LSB/dps). Python converts.
// Sensors sleep until a BLE client connects, and sleep again on disconnect.

#include <Arduino.h>
#include <Wire.h>
#include <NimBLEDevice.h>
#include "MAX30105.h"

#define DEVICE_NAME   "CardioGlasses"
#define SERVICE_UUID  "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define DATA_UUID     "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

#define MPU_ADDR 0x68
static const uint8_t LED_POWER = 0x7F;      // your working value
static const int SAMPLES_PER_PACKET = 4;    // 4 x 24 B = 96 B per notification
static const int SAMPLE_BYTES = 24;

MAX30105 ppg;
NimBLEServer* server = nullptr;
NimBLECharacteristic* dataChar = nullptr;

uint8_t pkt[SAMPLES_PER_PACKET * SAMPLE_BYTES];
int pktCount = 0;
uint32_t sampleIdx = 0;
bool streaming = false;

static void put32(uint8_t* p, uint32_t v) {
  p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24;
}
static void put16(uint8_t* p, int16_t v) {
  p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF;
}

// ---------- MPU6050 (raw registers, no library) ----------
static bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static uint8_t mpuRead8(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 1);
  return Wire.available() ? Wire.read() : 0xFF;
}

void setupIMU() {
  if (!mpuWrite(0x6B, 0x00)) {          // PWR_MGMT_1: wake up
    Serial.println("MPU6050 not found - check wiring");
    while (true) delay(1000);
  }
  delay(50);
  mpuWrite(0x1A, 0x03);                 // DLPF ~44 Hz
  mpuWrite(0x1B, 0x08);                 // gyro +-500 dps
  mpuWrite(0x1C, 0x08);                 // accel +-4 g
  Serial.printf("MPU6050 ready (WHO_AM_I=0x%02X)\n", mpuRead8(0x75));
  mpuWrite(0x6B, 0x40);                 // sleep until a client connects
}

// Reads accel + gyro (skips temperature) into out[6]
static void readIMU(int16_t out[6]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14);
  uint8_t b[14] = {0};
  for (int i = 0; i < 14 && Wire.available(); i++) b[i] = Wire.read();
  out[0] = (b[0] << 8) | b[1];   // ax
  out[1] = (b[2] << 8) | b[3];   // ay
  out[2] = (b[4] << 8) | b[5];   // az
  out[3] = (b[8] << 8) | b[9];   // gx
  out[4] = (b[10] << 8) | b[11]; // gy
  out[5] = (b[12] << 8) | b[13]; // gz
}

// ---------- MAX30102 ----------
void setupPPG() {
  if (!ppg.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found - check wiring");
    while (true) delay(1000);
  }
  // ledPower, sampleAverage, ledMode(2=Red+IR), sampleRate, pulseWidth, adcRange
  ppg.setup(LED_POWER, 4, 2, 400, 411, 16384);  // 100 Hz output
  ppg.shutDown();
  Serial.println("MAX30102 ready");
}

// ---------- BLE ----------
void setupBLE() {
  NimBLEDevice::init(DEVICE_NAME);
  NimBLEDevice::setMTU(247);
  server = NimBLEDevice::createServer();
  server->advertiseOnDisconnect(true);
  NimBLEService* svc = server->createService(SERVICE_UUID);
  dataChar = svc->createCharacteristic(DATA_UUID, NIMBLE_PROPERTY::NOTIFY);
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(DEVICE_NAME);
  adv->addServiceUUID(SERVICE_UUID);
  adv->enableScanResponse(true);
  adv->start();
  Serial.println("BLE advertising as " DEVICE_NAME);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  setupPPG();
  setupIMU();
  setupBLE();
}

void loop() {
  bool connected = server->getConnectedCount() > 0;

  if (connected && !streaming) {
    mpuWrite(0x6B, 0x00);   // wake IMU
    ppg.wakeUp();
    ppg.clearFIFO();
    sampleIdx = 0;
    pktCount = 0;
    streaming = true;
    Serial.println("Client connected - sensors ON");
  }
  if (!connected && streaming) {
    ppg.shutDown();
    mpuWrite(0x6B, 0x40);   // sleep IMU
    streaming = false;
    Serial.println("Client disconnected - sensors OFF");
  }
  if (!streaming) { delay(50); return; }

  ppg.check();
  while (ppg.available()) {
    uint32_t ir = ppg.getFIFOIR();
    uint32_t red = ppg.getFIFORed();
    ppg.nextSample();

    int16_t imu[6];
    readIMU(imu);           // sampled alongside each PPG sample

    uint8_t* p = pkt + pktCount * SAMPLE_BYTES;
    put32(p, sampleIdx++);
    put32(p + 4, ir);
    put32(p + 8, red);
    for (int i = 0; i < 6; i++) put16(p + 12 + 2 * i, imu[i]);

    if (++pktCount == SAMPLES_PER_PACKET) {
      dataChar->setValue(pkt, sizeof(pkt));
      dataChar->notify();
      pktCount = 0;
    }
  }
}