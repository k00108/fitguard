// =============================================================================
//  bt_imu_proto.hpp —— FitGuard 蓝牙 IMU 帧协议的【唯一权威实现】（C++ 接口）
//  ---------------------------------------------------------------------------
//  谁在用：
//    · bt_imu_serial.cpp    —— 从 USB 串口（桥模块 / USB-TTL）收数据
//    · bt_imu_uartlite.cpp  —— 从 PL 的 axi_uartlite 收数据（PYNQ 上，mmap /dev/mem）
//
//  帧格式（与 esp32/bt_imu_node/bt_imu_node.ino 严格一致）：
//     [0]=0xAA [1]=0x55 [2]=type [3]=len [4..3+len]=payload [4+len]=crc_lo [5+len]=crc_hi
//     CRC16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)，范围 = type,len,payload
//     type 0x01 单条记录 / 0x10 批量 / 0x02 PING / 0x03 PONG
//     记录 19 字节：seq(u8) t_us(u32 LE) ax ay az gx gy gz temp (7 × int16 LE)
//
//  平台：POSIX（Linux / WSL）。PYNQ 的 g++ 直接可用。
// =============================================================================
#ifndef FITGUARD_BT_IMU_PROTO_HPP
#define FITGUARD_BT_IMU_PROTO_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fitguard {

// ---------------------------------------------------------------- 协议常量 --
constexpr uint8_t  SYNC0         = 0xAA;
constexpr uint8_t  SYNC1         = 0x55;
constexpr uint8_t  TYPE_IMU      = 0x01;
constexpr uint8_t  TYPE_IMU_BATCH= 0x10;
constexpr uint8_t  TYPE_PING     = 0x02;
constexpr uint8_t  TYPE_PONG     = 0x03;
constexpr size_t   REC_LEN       = 19;   // 单条记录字节数
constexpr size_t   MAX_PAYLOAD   = 190;  // 单帧 payload 上限（= 10 条记录）

// 固件里的量程设置：AFS_SEL=0 → ±2 g；FS_SEL=1 → ±500 dps
constexpr double   ACCEL_LSB_PER_G  = 16384.0;
constexpr double   GYRO_LSB_PER_DPS = 65.5;

// ---------------------------------------------------------------- 协议函数 --
// 与固件 crc16() / Python binascii.crc_hqx(data, 0xFFFF) 逐位一致
uint16_t crc16_ccitt_false(const uint8_t* data, std::size_t n);

// 组帧：[AA][55][type][len][payload][crc_lo][crc_hi]
std::vector<uint8_t> build_frame(uint8_t type, const uint8_t* payload, std::size_t plen);

// 主机单调时钟（微秒）。等价于 Python 的 perf_counter_ns()//1000
int64_t host_us();

// ------------------------------------------------------------------ 数据结构 --
struct Record {
    int64_t  host_us = 0;   // 主机收到该帧的时刻
    uint8_t  seq     = 0;
    uint32_t t_us    = 0;   // 节点（穿戴端）采样时刻
    int16_t  ax = 0, ay = 0, az = 0;
    int16_t  gx = 0, gy = 0, gz = 0;
    int16_t  temp = 0;
};

struct Pong {
    uint8_t  id = 0;
    int64_t  t0 = 0;   // 主机发出 PING
    uint32_t t1 = 0;   // 节点收到
    uint32_t t2 = 0;   // 节点发出
    int64_t  t3 = 0;   // 主机收到 PONG
};

struct Sync {
    bool    valid         = false;
    int64_t rtt_us        = 0;   // 往返时间（已扣掉节点处理时间）
    double  offset_us     = 0.0; // 节点时钟 - 主机时钟
    double  one_way_us    = 0.0;
    double  jitter_std_us = 0.0; // 偏移估计的抖动 ← 答辩里的“同步窗口”
    std::size_t n_pong    = 0;
};

// ------------------------------------------------------------------- 解析器 --
class FrameParser {
public:
    // 喂入任意长度的字节流
    void feed(const uint8_t* data, std::size_t n);

    // 生成一个 PING 帧，并记下 t0（host_us）
    // out_id 非空时回传本次 ping 的 id，便于发送完成后用 restamp_ping 修正 t0
    std::vector<uint8_t> make_ping(uint8_t* out_id = nullptr);

    // 发送真正完成后（例如 TX FIFO 已空）修正 t0，提高时钟对齐精度
    void restamp_ping(uint8_t id, int64_t t0);

    // ---- 统计计数器（答辩时直接展示链路质量）----
    std::size_t n_bytes()   const { return n_bytes_;   }
    std::size_t n_frames()  const { return n_frames_;  }
    std::size_t n_bad_crc() const { return n_bad_crc_; }
    std::size_t n_bad_len() const { return n_bad_len_; }
    const std::vector<Record>& records() const { return records_; }
    const std::vector<Pong>&   pongs()   const { return pongs_;   }

    // NTP 四时间戳法，取 RTT 最小的那次作为最优估计
    Sync best_sync() const;

    // 打印一套可以直接抄进答辩 PPT 的指标
    void report() const;

    // 导出 CSV，返回写入行数
    std::size_t write_csv(const std::string& path) const;

private:
    void handle(uint8_t type, const uint8_t* payload, std::size_t len, int64_t rx_us);

    std::vector<uint8_t> buf_;
    std::size_t n_bytes_   = 0;
    std::size_t n_frames_  = 0;
    std::size_t n_bad_crc_ = 0;
    std::size_t n_bad_len_ = 0;
    std::vector<Record> records_;
    std::vector<Pong>   pongs_;
    struct Pend { uint8_t id; int64_t t0; };
    std::vector<Pend> pending_;
    uint8_t ping_id_ = 0;
};

}  // namespace fitguard

#endif  // FITGUARD_BT_IMU_PROTO_HPP
