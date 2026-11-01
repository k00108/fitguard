/* =============================================================================
 *  FitGuard 穿戴端 IMU 节点固件（ESP32 + MPU6050）
 *  ---------------------------------------------------------------------------
 *  作用：200 Hz 采集 MPU6050 六轴 + 温度，打成带【序号 + 节点本地时间戳 + CRC16】
 *        的二进制帧，通过蓝牙发给板端接收模块。
 *
 *  三种传输模式（编译期选择 TRANSPORT）：
 *    1) TRANSPORT_BLE  : BLE + Nordic UART Service (NUS)，未连接时自动广播。
 *                        —— “蓝牙”叙事用这个，配合 PL 侧字节级时间戳。
 *    2) TRANSPORT_SPP  : 经典蓝牙串口 (SPP/SerialBT)。
 *                        —— 带宽最大、延迟最低，比赛现场最稳。
 *    3) TRANSPORT_ESPNOW: ESP-NOW（板端也用一块 ESP32 收）。
 *                        —— 没有连接间隔约束，端到端延迟最低，答辩时可当“无线保底”。
 *
 *  收发双向：板端可以发 PING，本节点回 PONG，用于 NTP 式时钟对齐
 *            （这是回答评委“蓝牙延迟 / 同步窗口怎么解决”的核心机制）。
 *
 *  帧格式（所有类型统一，接收端一个状态机就能解）：
 *      [0] 0xAA  [1] 0x55  [2] type  [3] len
 *      [4 .. 3+len] payload
 *      [4+len] crc_lo  [5+len] crc_hi
 *      CRC16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)，
 *              计算范围 = type, len 以及全部 payload 字节（不含 0xAA55 和 CRC 本身）
 *
 *      type 0x01 : payload = 1 条 IMU 记录（19 字节）
 *      type 0x10 : payload = N 条 IMU 记录（19N 字节，批量打包，BLE 必用）
 *      type 0x02 : payload = ping_id(1) + host_t_us(4)                    (5 B)  板端 → 本节点
 *      type 0x03 : payload = ping_id(1) + host_t_us(4) + node_rx_us(4) + node_tx_us(4) (13 B) 本节点 → 板端
 *
 *  单条 IMU 记录（19 字节，小端）：
 *      seq(u8) t_us(u32) ax ay az gx gy gz temp (7 × int16)
 *
 *  依赖：Arduino-ESP32 core（board = "ESP32 Dev Module"），无需额外库。
 * ========================================================================== */

#include <Arduino.h>
#include <Wire.h>

// ----------------------------- 用户配置 ------------------------------------
#define TRANSPORT           TRANSPORT_BLE     // TRANSPORT_BLE / TRANSPORT_SPP / TRANSPORT_ESPNOW
#define ODR_HZ              200               // 采样率：100 / 200
#define BATCH_N             5                 // 每种传输一包塞几条记录（BLE 必须 >1）
#define BATCH_TIMEOUT_US    10000             // 凑不满 BATCH_N 也最多等这么久就发（控延迟）

// ---------------- 有线直连自检（强烈建议第一次上电就用它）------------------
// 置 1 后：把每一帧【同时】从 USB 串口吐出去，并且不再要求蓝牙已连接。
// 用法：ESP32 用 USB 线插到 PYNQ（或 Linux 电脑）上，
//       在那台机器上跑 bt_imu_serial，就能在【完全不用蓝牙、不用桥、不改 BD】的
//       情况下验证：传感器固件 + 帧协议 + CRC + 解析程序 + 时间戳质量。
// 顺便还验证了"PYNQ 能不能认出这块 ESP32 的 USB 串口"——正是方案 A 的前提。
// 波特率会自动提到 921600（115200 装不下 200 Hz × 25 B 并叠加文本日志）。
// 说明：镜像出去的流里混着二进制帧和文本日志，解析程序会自动跳过文本
//       （0xAA 不是可打印 ASCII，不会造成误同步）。
#define MIRROR_TO_USB       0
#define DEVICE_NAME         "FitGuard-IMU"
#define MPU_ADDR            0x68
#define I2C_SDA             21
#define I2C_SCL             22
#define I2C_HZ              400000

#define TRANSPORT_BLE       1
#define TRANSPORT_SPP       2
#define TRANSPORT_ESPNOW    3

// 板端 ESP32 的 MAC（只有 TRANSPORT_ESPNOW 需要改这里）
static uint8_t ESPNOW_PEER[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ----------------------------- 协议常量 ------------------------------------
#define SYNC0               0xAA
#define SYNC1               0x55
#define TYPE_IMU            0x01
#define TYPE_IMU_BATCH      0x10
#define TYPE_PING           0x02
#define TYPE_PONG           0x03
#define REC_LEN             19
#define MAX_FRAME           256

// ----------------------------- 下行字节环形缓冲 ----------------------------
// 板端发来的字节（PING）先入环形缓冲，主循环统一解析，避免在 BLE 回调里干活。
#define HOST_RB_SIZE 256
static volatile uint8_t hostRB[HOST_RB_SIZE];
static volatile uint16_t hostHead = 0, hostTail = 0;

static inline void hostPush(uint8_t b) {
  uint16_t n = (uint16_t)((hostHead + 1) % HOST_RB_SIZE);
  if (n == hostTail) return;                 // 满则丢
  hostRB[hostHead] = b;
  hostHead = n;
}
static inline int hostPop() {
  if (hostTail == hostHead) return -1;
  uint8_t b = hostRB[hostTail];
  hostTail = (uint16_t)((hostTail + 1) % HOST_RB_SIZE);
  return b;
}

// ----------------------------- 传输层 --------------------------------------
#if TRANSPORT == TRANSPORT_BLE
  #include <BLEDevice.h>
  #include <BLEServer.h>
  #include <BLEUtils.h>
  #include <BLE2902.h>
  #include <esp_gap_ble_api.h>
  #include <esp_gatt_common_api.h>
  // 注意：下面 onConnect 的第二个参数类型在 Arduino-ESP32 core 2.x 里是
  //       esp_ble_gatts_cb_param_t*，3.x 里是 esp_ble_gatt_conn_params_t*。
  //       本文件按 3.x 写；用 2.x 的话把类型换掉、并把 p->remote_bda 改成
  //       param->connect.remote_bda 即可。

  #define NUS_SERVICE "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
  #define NUS_RX_CHAR "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // 中心设备写 → 本节点
  #define NUS_TX_CHAR "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // 本节点 notify → 中心设备

  static BLEServer         *bleServer = nullptr;
  static BLECharacteristic *bleTx     = nullptr;
  static volatile bool      bleConnected = false;
  static volatile bool      bleNotifyOn  = false;
  static esp_bd_addr_t      blePeer;

  class ServerCB : public BLEServerCallbacks {
    void onConnect(BLEServer *s, esp_ble_gatt_conn_params_t *p) override {
      bleConnected = true;
      memcpy(blePeer, p->remote_bda, 6);
      // 关键：缩短连接间隔，否则 200 Hz 数据会被连接事件“卡”住。
      // 参数单位：interval 1.25 ms，timeout 10 ms。6/12 = 7.5~15 ms。
      esp_ble_gap_set_prefer_conn_params(blePeer, 6, 12, 0, 400);
      // 关键：协商大 MTU，否则一次只能发 20 字节，批量打包就废了。
      esp_ble_gatt_set_local_mtu(247);
    }
    void onDisconnect(BLEServer *s) override {
      bleConnected = false;
      bleNotifyOn  = false;
      s->startAdvertising();          // 断线自动重新广播
    }
  };

  class TxCB : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *c) override {
      String v = c->getValue();
      for (size_t i = 0; i < v.length(); i++) hostPush((uint8_t)v[i]);
    }
  };

  // 中心设备写 CCCD（0x2902）打开 notify 后，才允许发数据——否则 notify 会丢。
  class CccdCB : public BLEDescriptorCallbacks {
    void onWrite(BLEDescriptor *d) override {
      uint8_t *v = d->getValue();
      if (v && d->getLength() >= 1) bleNotifyOn = (v[0] & 0x01) != 0;
    }
  };

  static void transportBegin() {
    BLEDevice::init(DEVICE_NAME);
    BLEDevice::setMTU(247);
    // 穿在身上时人体对 2.4 GHz 遮挡很严重，发射功率直接拉满
    BLEDevice::setPower(ESP_PWR_LVL_P9);
    bleServer = BLEDevice::createServer();
    bleServer->setCallbacks(new ServerCB());

    BLEService *svc = bleServer->createService(NUS_SERVICE);
    bleTx = svc->createCharacteristic(NUS_TX_CHAR, BLECharacteristic::PROPERTY_NOTIFY);
    BLE2902 *cccd = new BLE2902();
    cccd->setCallbacks(new CccdCB());
    bleTx->addDescriptor(cccd);
    BLECharacteristic *rx = svc->createCharacteristic(NUS_RX_CHAR, BLECharacteristic::PROPERTY_WRITE);
    rx->setCallbacks(new TxCB());
    svc->start();

    BLEAdvertising *adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(NUS_SERVICE);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);       // 让中心设备倾向 7.5 ms 连接间隔
    adv->setMaxPreferred(0x0C);
    BLEDevice::startAdvertising();
  }

  static bool transportReady()   { return bleConnected && bleNotifyOn; }
  static bool transportSend(const uint8_t *d, size_t n) {
    if (!bleConnected) return false;
    bleTx->setValue((uint8_t *)d, n);
    bleTx->notify();
    return true;
  }
  static size_t transportMaxPayload() { return 244; }   // MTU 247 - 3
  static void transportPoll() { /* BLE 回调驱动，这里不用做事 */ }
  static int  transportRead() { return hostPop(); }

#elif TRANSPORT == TRANSPORT_SPP
  #include <BluetoothSerial.h>
  static BluetoothSerial SerialBT;
  static void transportBegin() { SerialBT.begin(DEVICE_NAME); }
  static bool transportReady()   { return SerialBT.hasClient(); }
  static bool transportSend(const uint8_t *d, size_t n) { return SerialBT.write(d, n) == n; }
  static size_t transportMaxPayload() { return 240; }
  static void transportPoll() { /* 收到的主机字节在 loop 里 poll */ }
  static int  transportRead() { return SerialBT.available() ? SerialBT.read() : -1; }

#else   // TRANSPORT_ESPNOW
  #include <WiFi.h>
  #include <esp_now.h>
  static volatile bool espnowReady = false;
  static void onEspnowSent(const uint8_t *mac, esp_now_send_status_t st) {}
  static void onEspnowRecv(const uint8_t *mac, const uint8_t *data, int len) {
    for (int i = 0; i < len; i++) hostPush(data[i]);
  }
  static void transportBegin() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_19_5dBm);   // 穿在身上，功率拉满
    if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW init fail"); return; }
    esp_now_register_send_cb(onEspnowSent);
    esp_now_register_recv_cb(onEspnowRecv);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, ESPNOW_PEER, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
    espnowReady = true;
  }
  static bool transportReady() { return espnowReady; }
  static bool transportSend(const uint8_t *d, size_t n) {
    return esp_now_send(ESPNOW_PEER, d, n) == ESP_OK;   // 上限 250 字节
  }
  static size_t transportMaxPayload() { return 240; }
  static void transportPoll() {}
  static int  transportRead() { return hostPop(); }
#endif

// ----------------------------- MPU6050 寄存器级驱动 ------------------------
#define REG_SMPLRT_DIV    0x19
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C
#define REG_ACCEL_XOUT_H  0x3B
#define REG_PWR_MGMT_1    0x6B
#define REG_WHO_AM_I      0x75

static void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_WHO_AM_I);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 1);
  if (!Wire.available()) return false;
  uint8_t who = Wire.read();
  if (who != 0x68 && who != 0x70 && who != 0x71 && who != 0x73) return false;

  mpuWrite(REG_PWR_MGMT_1, 0x00);      // 唤醒，内部 8 MHz 时钟
  delay(50);
  mpuWrite(REG_PWR_MGMT_1, 0x01);      // 用 PLL X 轴陀螺作时钟源，更稳
  delay(10);
  // DLPF_CFG = 3 → 陀螺带宽 42 Hz / 加速度计 44 Hz，内部采样 1 kHz
  mpuWrite(REG_CONFIG, 0x03);
  // SMPLRT_DIV = 1000/ODR - 1 → 200 Hz 时 = 4
  mpuWrite(REG_SMPLRT_DIV, (uint8_t)(1000 / ODR_HZ - 1));
  mpuWrite(REG_GYRO_CONFIG, 0x08);     // ±500 dps  → 65.5 LSB/dps
  mpuWrite(REG_ACCEL_CONFIG, 0x00);    // ±2 g      → 16384 LSB/g
  return true;
}

// 一次突发读 14 字节：ax ay az temp gx gy gz
static bool mpuReadRaw(int16_t out[7]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();
  out[0] = (int16_t)((b[0]  << 8) | b[1]);    // ax
  out[1] = (int16_t)((b[2]  << 8) | b[3]);    // ay
  out[2] = (int16_t)((b[4]  << 8) | b[5]);    // az
  out[3] = (int16_t)((b[6]  << 8) | b[7]);    // temp (原始值，/340 + 36.53 °C)
  out[4] = (int16_t)((b[8]  << 8) | b[9]);    // gx
  out[5] = (int16_t)((b[10] << 8) | b[11]);   // gy
  out[6] = (int16_t)((b[12] << 8) | b[13]);   // gz
  return true;
}

// ----------------------------- CRC16 / CCITT-FALSE ------------------------
// 与 Python 的 binascii.crc_hqx(data, 0xFFFF) 完全一致
static uint16_t crc16(const uint8_t *d, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (int k = 0; k < 8; k++)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

// ----------------------------- 组帧缓冲区 ----------------------------------
static uint8_t  txbuf[MAX_FRAME];
static uint8_t  recBuf[REC_LEN * 32];
static int      recCount = 0;
static uint8_t  txseq = 0;

// 把 payload 帧头/CRC 补上，放进 txbuf，返回整帧长度
static size_t buildFrame(uint8_t type, const uint8_t *payload, size_t plen) {
  txbuf[0] = SYNC0;
  txbuf[1] = SYNC1;
  txbuf[2] = type;
  txbuf[3] = (uint8_t)plen;
  memcpy(&txbuf[4], payload, plen);
  uint16_t c = crc16(&txbuf[2], plen + 2);       // type, len, payload
  txbuf[4 + plen] = (uint8_t)(c & 0xFF);
  txbuf[5 + plen] = (uint8_t)(c >> 8);
  return 6 + plen;
}

static void packRec(uint8_t *dst, uint8_t seq, uint32_t t_us, const int16_t v[7]) {
  dst[0] = seq;
  dst[1] = (uint8_t)(t_us);
  dst[2] = (uint8_t)(t_us >> 8);
  dst[3] = (uint8_t)(t_us >> 16);
  dst[4] = (uint8_t)(t_us >> 24);
  for (int i = 0; i < 7; i++) {
    dst[5 + 2 * i] = (uint8_t)(v[i] & 0xFF);
    dst[6 + 2 * i] = (uint8_t)((v[i] >> 8) & 0xFF);
  }
}

// ----------------------------- 主机下行帧解析（PING → PONG）----------------
static uint8_t  rxstate = 0;
static uint8_t  rxbuf[64];
static uint8_t  rxlen = 0, rxidx = 0, rxtype = 0;

static void handleHostFrame(uint8_t type, uint8_t *pl, uint8_t len) {
  if (type == TYPE_PING && len >= 5) {
    // NTP 式的四时间戳：host 发 t0 → node 收 t1 → node 发 t2 → host 收 t3
    // host 侧偏移 = ((t1 - t0) + (t2 - t3)) / 2，单程延迟 = ((t3 - t0) - (t2 - t1)) / 2
    uint32_t node_rx = (uint32_t)esp_timer_get_time();
    uint8_t p[13];
    p[0] = pl[0];                              // ping_id 原样回
    memcpy(&p[1], &pl[1], 4);                  // host_t_us 原样回
    uint32_t node_tx = (uint32_t)esp_timer_get_time();
    memcpy(&p[5], &node_rx, 4);
    memcpy(&p[9], &node_tx, 4);
    size_t n = buildFrame(TYPE_PONG, p, 13);
    transportSend(txbuf, n);
  }
}

static void parseHostByte(uint8_t b) {
  switch (rxstate) {
    case 0: if (b == SYNC0) rxstate = 1; break;
    case 1: rxstate = (b == SYNC1) ? 2 : (b == SYNC0 ? 1 : 0); break;
    case 2: rxtype = b; rxstate = 3; break;
    case 3: rxlen = b; rxidx = 0; rxstate = (b > 0 && b <= sizeof(rxbuf)) ? 4 : 0; break;
    case 4:
      rxbuf[rxidx++] = b;
      if (rxidx >= rxlen) rxstate = 5;
      break;
    case 5: rxbuf[rxidx] = b; rxstate = 6; break;   // 用 rxbuf 末尾两字节暂存 CRC
    case 6: {
      uint16_t got = (uint16_t)rxbuf[rxlen] | ((uint16_t)b << 8);
      // 重新算 CRC：type, len, payload
      uint8_t tmp[66];
      tmp[0] = rxtype; tmp[1] = rxlen;
      memcpy(&tmp[2], rxbuf, rxlen);
      if (crc16(tmp, rxlen + 2) == got) handleHostFrame(rxtype, rxbuf, rxlen);
      rxstate = 0;
      break;
    }
  }
}

// ----------------------------- 主循环 --------------------------------------
static uint32_t next_us = 0;
static uint32_t batchStart_us = 0;      // 本批第一条记录的时刻，用于超时强制发包
static uint32_t sentFrames = 0, dropFrames = 0;

// 三种传输都提供了 transportRead()：SPP 直接读 SerialBT，BLE / ESP-NOW 从环形缓冲取
#define HOST_READ() transportRead()

// 是否允许发包：开了有线自检就永远允许（不要求蓝牙已连接）
static inline bool tx_allowed() {
#if MIRROR_TO_USB
  return true;
#else
  return transportReady();
#endif
}

void setup() {
  Serial.begin(MIRROR_TO_USB ? 921600 : 115200);
  delay(200);
  Wire.begin(I2C_SDA, I2C_SCL, I2C_HZ);

  if (!mpuInit()) {
    Serial.println("[FATAL] MPU6050 not found. Check SDA/SCL and AD0 (addr 0x68/0x69).");
    while (1) delay(1000);
  }
  Serial.printf("[OK] MPU6050 init, ODR=%d Hz, batch=%d\n", ODR_HZ, BATCH_N);

  transportBegin();
  Serial.printf("[OK] transport=%d, name=%s\n", TRANSPORT, DEVICE_NAME);

  next_us = (uint32_t)esp_timer_get_time();
}

void loop() {
  // ---- 1) 采一帧 ----
  uint32_t now = (uint32_t)esp_timer_get_time();
  if ((int32_t)(now - next_us) >= 0) {
    next_us += 1000000UL / ODR_HZ;

    // 如果落后超过 20 ms，说明被阻塞了：重新对时并记一次丢样
    if ((int32_t)(now - next_us) > 20000) {
      dropFrames++;
      next_us = now + 1000000UL / ODR_HZ;
    }

    int16_t v[7];
    // 时间戳要尽量贴近 I2C 读取的时刻：放在读之前，抖动远小于“采样周期”本身
    uint32_t t_sample = (uint32_t)esp_timer_get_time();
    if (mpuReadRaw(v)) {
      if (recCount == 0) batchStart_us = t_sample;
      packRec(&recBuf[recCount * REC_LEN], txseq++, t_sample, v);
      recCount++;
    }
  }

  // ---- 2) 决定是否发包 ----
  bool flush = false;
  if (recCount >= BATCH_N) flush = true;                       // 攒够一包
  if (recCount > 0 &&
      (uint32_t)((uint32_t)esp_timer_get_time() - batchStart_us) >= BATCH_TIMEOUT_US) {
    flush = true;                                              // 超时，先发出去降延迟
  }

  if (flush && recCount > 0 && tx_allowed()) {
    size_t plen = (size_t)recCount * REC_LEN;
    uint8_t type = (recCount == 1) ? TYPE_IMU : TYPE_IMU_BATCH;
    size_t n = buildFrame(type, recBuf, plen);
    if (n <= transportMaxPayload() + 6) {
      if (transportReady()) {
        if (!transportSend(txbuf, n)) dropFrames++;
      }
#if MIRROR_TO_USB
      Serial.write(txbuf, n);             // 有线自检：同一帧原样从 USB 吐出去
#endif
      sentFrames++;
    } else {
      dropFrames++;                       // 一包太大，减小 BATCH_N
    }
    recCount = 0;
  }
  if (!tx_allowed()) recCount = 0;        // 未连接时不堆积

  // ---- 3) 收板端下行（PING）----
  int b;
  while ((b = HOST_READ()) >= 0) parseHostByte((uint8_t)b);

  transportPoll();

  // ---- 4) 心跳打印 ----
  static uint32_t tLast = 0;
  if (millis() - tLast > 2000) {
    tLast = millis();
    Serial.printf("[stat] sent=%lu drop=%lu rec=%d conn=%d\n",
                  (unsigned long)sentFrames, (unsigned long)dropFrames,
                  recCount, (int)transportReady());
  }
}
