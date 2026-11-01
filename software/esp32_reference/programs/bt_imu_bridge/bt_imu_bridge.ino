/* =============================================================================
 *  bt_imu_bridge.ino —— 外挂蓝牙网关（放在 PYNQ 外面）
 *  ---------------------------------------------------------------------------
 *  角色：这是"无线 → 有线"的转换器。
 *        它用无线（BLE / 经典蓝牙 SPP / ESP-NOW）收穿戴端的数据，
 *        然后【原样】从 USB 串口吐给 PYNQ。
 *
 *  为什么要外挂：
 *    PYNQ-Z2 板子上没有蓝牙无线电，而且官方 PYNQ 镜像没编 `btusb` 内核模块
 *    （社区 2020→2025 反复确认，无可用解法）。把网关放在板子外面，
 *    PYNQ 侧只需要认一个普通 USB 串口设备 /dev/ttyUSB0 ——
 *    【不改 BD、不重新综合、不动 fitguard_unified.bit、不碰你们已验证的视觉链路】。
 *
 *  接线（外挂网关 → PYNQ）：
 *      ［网关 ESP32］── USB 线 ──▶ ［PYNQ 的 USB-A］
 *      USB 一根线同时提供【供电 + 数据】，网关不需要额外电源。
 *      ⚠️ PYNQ 只有 1 个 USB-A 主机口，而摄像头占着它 → 需要带独立供电的 USB Hub。
 *         验证：在 PYNQ 上 `lsusb`，应能看到 Hub + 摄像头 + 本网关（CH340/CP2102/cdc_acm）。
 *
 *  PYNQ 侧怎么用：
 *      sudo ./cpp/bt_imu_serial --list
 *      sudo ./cpp/bt_imu_serial --port /dev/ttyUSB0 --baud 921600 --duration 10
 *      帧格式不用动：网关是"透传"，字节流和有线时一模一样。
 *
 *  双向能力（时钟对齐要用）：
 *      PYNQ → 网关 → 穿戴端：把 PYNQ 发的 PING 帧转发回去（走 NUS RX 特征 / SPP / ESP-NOW）
 *      穿戴端 → 网关 → PYNQ：PONG 原样吐给 PYNQ
 *      这样 `bt_imu_proto` 里的 NTP 式四时间戳就能工作，你们就有"同步窗口"这个指标。
 *
 *  目标环境：Arduino-ESP32 core 3.x（board = "ESP32 Dev Module"）
 *  ⚠️ core 2.x 的兼容改写点集中在文件末尾"2.x 兼容"注释块里，照改即可。
 *
 *  三种模式（编译期选 BRIDGE_MODE）：
 *      1 = BLE 中心设备（推荐，叙事上就是"蓝牙"）
 *      2 = 经典蓝牙 SPP 主设备（最稳、带宽最大）
 *      3 = ESP-NOW（无连接无配对，延迟最低）
 * ========================================================================== */

#include <Arduino.h>

// ------------------------------- 用户配置 ----------------------------------
#define BRIDGE_MODE_BLE      1
#define BRIDGE_MODE_SPP      2
#define BRIDGE_MODE_ESPNOW   3

#define BRIDGE_MODE          BRIDGE_MODE_BLE  // ← 改这里切换
#define NODE_NAME            "FitGuard-IMU"   // 必须和穿戴端 bt_imu_node.ino 的 DEVICE_NAME 一致
#define BRIDGE_BAUD          921600           // PYNQ 侧 bt_imu_serial 的 --baud 要一致
#define STATS_INTERVAL_MS    10000UL          // 每隔多久打一行状态（文本会被解析器自动跳过）
#define RECONNECT_COOLDOWN_MS 2000UL

// ESP-NOW 模式下要填穿戴端的 MAC；全 0xFF 表示广播
static uint8_t ESPNOW_PEER[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ------------------------------- 统计 --------------------------------------
static uint32_t statFwdDown = 0;   // 无线 → 串口 的字节数
static uint32_t statFwdUp   = 0;   // 串口 → 无线 的字节数
static uint32_t statFramesUp = 0;
static uint32_t statReconnects = 0;

// =========================== 下行环形缓冲（无线→串口）======================
// 绝对不要在无线回调里直接写 Serial：回调在蓝牙任务上下文里，一旦阻塞会丢包。
#define RB_SIZE 2048
static volatile uint8_t  rb[RB_SIZE];
static volatile uint16_t rbHead = 0, rbTail = 0;

static inline void rbPush(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint16_t next = (uint16_t)((rbHead + 1) % RB_SIZE);
    if (next == rbTail) {            // 满了：丢一字节并计数（正常不应发生）
      rbTail = (uint16_t)((rbTail + 1) % RB_SIZE);
    }
    rb[rbHead] = p[i];
    rbHead = next;
  }
}
static inline int rbPop() {
  if (rbTail == rbHead) return -1;
  uint8_t b = rb[rbTail];
  rbTail = (uint16_t)((rbTail + 1) % RB_SIZE);
  return b;
}
static inline size_t rbDrain(uint8_t* out, size_t cap) {
  size_t n = 0;
  int b;
  while (n < cap && (b = rbPop()) >= 0) out[n++] = (uint8_t)b;
  return n;
}

// =========================== 上行组帧（串口→无线）==========================
// PYNQ 发来的 PING 帧可能被 UART 分几次到达，这里攒齐一整帧再转发，
// 避免把一帧拆成两次 BLE 写导致节点收不全。
static uint8_t upBuf[64];
static size_t  upLen = 0;
static uint32_t upLastMs = 0;

static void forwardUp(const uint8_t* p, size_t n);   // 前向声明，三种模式各自实现

static void pumpHostToWireless() {
  while (Serial.available()) {
    uint8_t b = (uint8_t)Serial.read();
    if (upLen == 0) {
      if (b == 0xAA) upBuf[upLen++] = b;
      upLastMs = millis();
      continue;
    }
    if (upLen == 1) {
      if (b == 0x55) {
        upBuf[upLen++] = b;
      } else {
        upLen = 0;
        if (b == 0xAA) upBuf[upLen++] = b;
      }
      upLastMs = millis();
      continue;
    }
    if (upLen < sizeof(upBuf)) upBuf[upLen++] = b;
    upLastMs = millis();

    if (upLen >= 4) {
      const size_t need = 6 + upBuf[3];          // [AA][55][type][len][payload][crc16]
      if (need > sizeof(upBuf)) {                // 长度非法，丢弃重新同步
        upLen = 0;
      } else if (upLen >= need) {
        forwardUp(upBuf, need);
        statFramesUp++;
        upLen = 0;
      }
    }
  }
  // 半截帧超时丢弃，避免残留字节影响下一帧
  if (upLen > 0 && (millis() - upLastMs) > 100) upLen = 0;
}

// =========================== 模式 1：BLE 中心设备 ==========================
#if BRIDGE_MODE == BRIDGE_MODE_BLE
  #include <BLEDevice.h>
  #include <BLEServer.h>
  #include <BLEUtils.h>
  #include <esp_gatt_common_api.h>

  #define NUS_SERVICE "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
  #define NUS_RX_CHAR "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // 中心设备写 → 节点
  #define NUS_TX_CHAR "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // 节点 notify → 中心设备

  static BLEAdvertisedDevice* gTarget = nullptr;
  static BLERemoteCharacteristic* gUpChar = nullptr;    // 用来给节点下发（PING）
  static volatile bool gConnected = false;
  static bool gScanning = false;
  static uint32_t gDiscMs = 0;

  // 节点 notify 回调：只做入队，别干别的
  static void onNodeNotify(BLERemoteCharacteristic* ch, uint8_t* data, size_t len, bool isNotify) {
    rbPush(data, len);
  }

  class ClientCB : public BLEClientCallbacks {
    void onDisconnect(BLEClient* c) override {
      gConnected = false;
      gDiscMs = millis();
      Serial.println("### BLE 断开，准备重连");
    }
  };

  class ScanCB : public BLEAdvertisedDeviceCallbacks {
    void onResult(BLEAdvertisedDevice* dev) override {
      if (dev->haveName() && dev->getName() == NODE_NAME) {
        if (gTarget) delete gTarget;
        gTarget = new BLEAdvertisedDevice(*dev);
        BLEDevice::getScan()->stop();
      }
    }
  };

  static void forwardUp(const uint8_t* p, size_t n) {
    if (!gConnected || gUpChar == nullptr) return;
    gUpChar->writeValue(const_cast<uint8_t*>(p), n, false);
    statFwdUp += n;
  }

  static bool tryConnect() {
    if (gTarget == nullptr) return false;
    BLEClient* client = BLEDevice::getClient();
    if (client == nullptr) return false;
    client->setClientCallbacks(new ClientCB(), false);
    if (!client->connect(gTarget)) {
      Serial.println("### BLE connect 失败");
      return false;
    }
    // 关键：协商大 MTU。50 Hz × 25 B = 1.25 kB/s，默认 MTU 也够，
    //       但 MTU 247 能在突发时留足余量（节点已经按 5 条/包批量发）。
    esp_ble_gatt_set_local_mtu(247);

    BLERemoteService* svc = client->getService(BLEUUID(NUS_SERVICE));
    if (svc == nullptr) {
      Serial.println("### 找不到 NUS 服务");
      client->disconnect();
      return false;
    }
    BLERemoteCharacteristic* tx = svc->getCharacteristic(BLEUUID(NUS_TX_CHAR));
    gUpChar = svc->getCharacteristic(BLEUUID(NUS_RX_CHAR));
    if (tx == nullptr || gUpChar == nullptr) {
      Serial.println("### 找不到 NUS 特征");
      client->disconnect();
      return false;
    }
    if (!tx->canNotify()) {
      Serial.println("### TX 特征不支持 notify");
      client->disconnect();
      return false;
    }
    tx->registerForNotify(onNodeNotify);
    gConnected = true;
    Serial.printf("### BLE 已连接: %s\n", gTarget->getAddress().toString().c_str());
    return true;
  }

  static void beginWireless() {
    BLEDevice::init("FitGuard-Bridge");
    BLEDevice::setMTU(247);
    BLEDevice::setPower(ESP_PWR_LVL_P9);
    BLEScan* scan = BLEDevice::getScan();
    scan->setAdvertisedDeviceCallbacks(new ScanCB(), false);
    scan->setActiveScan(true);
    scan->setInterval(100);
    scan->setWindow(90);
  }

  static void pollWireless() {
    if (!gConnected) {
      if (!gScanning && (millis() - gDiscMs) > RECONNECT_COOLDOWN_MS) {
        Serial.println("### 开始扫描 " NODE_NAME);
        gScanning = true;
        BLEDevice::getScan()->start(3, false);   // 非阻塞，扫 3 秒
      }
      if (gScanning && gTarget != nullptr) {
        gScanning = false;
        if (tryConnect()) statReconnects++;
      }
      if (gScanning && (millis() - gDiscMs) > RECONNECT_COOLDOWN_MS + 4000) {
        gScanning = false;                       // 扫描窗口结束，下轮重来
      }
    }
  }

// =========================== 模式 2：经典蓝牙 SPP 主设备 ===================
#elif BRIDGE_MODE == BRIDGE_MODE_SPP
  #include <BluetoothSerial.h>
  static BluetoothSerial SerialBT;
  static bool gConnected = false;
  static uint32_t gDiscMs = 0;

  static void forwardUp(const uint8_t* p, size_t n) {
    if (!gConnected) return;
    SerialBT.write(p, n);
    statFwdUp += n;
  }

  static void beginWireless() {
    SerialBT.begin("FitGuard-Bridge", true);   // true = 主设备模式
  }

  static void pollWireless() {
    if (!gConnected) {
      if ((millis() - gDiscMs) > RECONNECT_COOLDOWN_MS) {
        gDiscMs = millis();
        Serial.println("### SPP 尝试连接 " NODE_NAME);
        if (SerialBT.connect(NODE_NAME)) {     // 按名字做一次 inquiry 再连
          gConnected = true;
          statReconnects++;
          Serial.println("### SPP 已连接");
        }
      }
      return;
    }
    if (!SerialBT.hasClient()) {
      gConnected = false;
      gDiscMs = millis();
      Serial.println("### SPP 断开");
      return;
    }
    // 无线 → 环形缓冲
    while (SerialBT.available()) {
      uint8_t b = (uint8_t)SerialBT.read();
      rbPush(&b, 1);
    }
  }

// =========================== 模式 3：ESP-NOW ==============================
#else
  #include <WiFi.h>
  #include <esp_now.h>
  static bool gReady = false;

  static void onEspnowRecv(const uint8_t* mac, const uint8_t* data, int len) {
    rbPush(data, (size_t)len);
  }

  static void forwardUp(const uint8_t* p, size_t n) {
    if (!gReady) return;
    esp_now_send(ESPNOW_PEER, p, n);
    statFwdUp += n;
  }

  static void beginWireless() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_19_5dBm);
    if (esp_now_init() != ESP_OK) {
      Serial.println("### ESP-NOW init 失败");
      return;
    }
    esp_now_register_recv_cb(onEspnowRecv);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, ESPNOW_PEER, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
    gReady = true;
  }

  static void pollWireless() { /* 回调驱动 */ }
#endif

// ================================= 主流程 ==================================
void setup() {
  Serial.begin(BRIDGE_BAUD);      // 这就是给 PYNQ 的那条线（USB-UART）
  delay(300);
  Serial.println();
  Serial.println("### FitGuard 外挂蓝牙网关启动");
  Serial.printf("### 模式=%d  波特率=%d  节点名=%s\n",
                BRIDGE_MODE, BRIDGE_BAUD, NODE_NAME);
  beginWireless();
}

void loop() {
  // 1) 无线 → PYNQ
  uint8_t buf[512];
  for (;;) {
    const size_t n = rbDrain(buf, sizeof(buf));
    if (n == 0) break;
    Serial.write(buf, n);
    statFwdDown += n;
  }

  // 2) 维护无线连接（扫描/重连/收数据）
  pollWireless();

  // 3) PYNQ → 无线（转发 PING，用于时钟对齐）
  pumpHostToWireless();

  // 4) 状态心跳（文本混在二进制流里没问题：0xAA 不是可打印 ASCII，解析器不会误同步）
  static uint32_t tLast = 0;
  if (millis() - tLast > STATS_INTERVAL_MS) {
    tLast = millis();
    Serial.printf("### 下行 %lu B  上行 %lu B / %lu 帧  重连 %lu 次\n",
                  (unsigned long)statFwdDown, (unsigned long)statFwdUp,
                  (unsigned long)statFramesUp, (unsigned long)statReconnects);
  }
}

/* =============================================================================
 *  Arduino-ESP32 core 2.x 兼容改写点（只有三处）
 *  ---------------------------------------------------------------------------
 *  1) 扫描回调签名：
 *       2.x:  void onResult(BLEAdvertisedDevice dev) override { ... }
 *             （按值传，内部用 dev.getName()、new BLEAdvertisedDevice(dev)）
 *  2) 创建客户端：
 *       2.x:  BLEClient* client = BLEDevice::createClient();
 *             用完要 BLEDevice::deleteClient(client);
 *       3.x:  BLEDevice::getClient()
 *  3) MTU：
 *       2.x:  BLEDevice::setMTU(247) 要在 init 之后、connect 之前调用
 *
 *  另外记录一条实测经验：BLE 中心设备的连接间隔由从设备发起请求最省事，
 *  穿戴端 bt_imu_node.ino 里已经调用了 esp_ble_gap_set_prefer_conn_params(6,12,...)，
 *  所以网关这边不用再设一遍。
 * ========================================================================== */
