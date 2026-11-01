// =============================================================================
//  bt_imu_proto.cpp —— FitGuard 蓝牙 IMU 帧协议实现（C++）
//  ---------------------------------------------------------------------------
//  与 esp32/bt_imu_node/bt_imu_node.ino 的组帧/CRC 逐位一致。
//  改协议只改这个文件和 .hpp，两个接收程序自动跟着变。
//
//  编译：
//     g++ -O2 -std=c++17 -Wall -Wextra -c bt_imu_proto.cpp -o bt_imu_proto.o
// =============================================================================
#include "bt_imu_proto.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace fitguard {

namespace {

double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

// 总体标准差（等价于 Python statistics.pstdev）
double stddev_of(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean_of(v);
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size()));
}

// 线性插值百分位（接近 numpy.percentile 的默认行为）
double percentile_of(std::vector<double> v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double pos = (pct / 100.0) * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return v[lo] + (v[hi] - v[lo]) * frac;
}

inline int16_t rd_i16(const uint8_t* p) {
    return static_cast<int16_t>(static_cast<uint16_t>(p[0]) |
                                (static_cast<uint16_t>(p[1]) << 8));
}

inline uint32_t rd_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

}  // namespace

// ---------------------------------------------------------------- 协议函数 --
uint16_t crc16_ccitt_false(const uint8_t* data, std::size_t n) {
    uint16_t crc = 0xFFFF;
    for (std::size_t i = 0; i < n; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (int k = 0; k < 8; ++k) {
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                                 : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

std::vector<uint8_t> build_frame(uint8_t type, const uint8_t* payload, std::size_t plen) {
    std::vector<uint8_t> f;
    f.reserve(6 + plen);
    f.push_back(SYNC0);
    f.push_back(SYNC1);
    f.push_back(type);
    f.push_back(static_cast<uint8_t>(plen));
    for (std::size_t i = 0; i < plen; ++i) f.push_back(payload[i]);
    const uint16_t c = crc16_ccitt_false(f.data() + 2, plen + 2);  // type,len,payload
    f.push_back(static_cast<uint8_t>(c & 0xFF));
    f.push_back(static_cast<uint8_t>(c >> 8));
    return f;
}

int64_t host_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// -------------------------------------------------------------- PING 生成 --
std::vector<uint8_t> FrameParser::make_ping(uint8_t* out_id) {
    ping_id_ = static_cast<uint8_t>(ping_id_ + 1);
    const int64_t t0 = host_us();
    if (out_id) *out_id = ping_id_;

    pending_.push_back(Pend{ping_id_, t0});
    // 只保留最近 32 个，防止一直没回包时无限增长
    if (pending_.size() > 32) pending_.erase(pending_.begin());

    uint8_t pl[5];
    pl[0] = ping_id_;
    pl[1] = static_cast<uint8_t>(t0 & 0xFF);
    pl[2] = static_cast<uint8_t>((t0 >> 8) & 0xFF);
    pl[3] = static_cast<uint8_t>((t0 >> 16) & 0xFF);
    pl[4] = static_cast<uint8_t>((t0 >> 24) & 0xFF);
    return build_frame(TYPE_PING, pl, sizeof(pl));
}

void FrameParser::restamp_ping(uint8_t id, int64_t t0) {
    for (auto it = pending_.rbegin(); it != pending_.rend(); ++it) {
        if (it->id == id) {
            it->t0 = t0;
            return;
        }
    }
}

// ---------------------------------------------------------------- 接收解析 --
void FrameParser::feed(const uint8_t* data, std::size_t n) {
    if (data == nullptr || n == 0) return;
    n_bytes_ += n;
    buf_.insert(buf_.end(), data, data + n);

    for (;;) {
        // 1) 找同步字 AA 55
        std::size_t sync_at = buf_.size();
        for (std::size_t k = 0; k + 1 < buf_.size(); ++k) {
            if (buf_[k] == SYNC0 && buf_[k + 1] == SYNC1) {
                sync_at = k;
                break;
            }
        }
        if (sync_at == buf_.size()) {
            // 没找到：保留最后 1 字节（可能是 0xAA），其余丢弃
            if (buf_.size() > 1) buf_.erase(buf_.begin(), buf_.end() - 1);
            return;
        }
        if (sync_at > 0) buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(sync_at));
        if (buf_.size() < 4) return;

        // 2) 长度检查
        const uint8_t ftype = buf_[2];
        const uint8_t flen  = buf_[3];
        if (flen == 0 || flen > MAX_PAYLOAD) {
            ++n_bad_len_;
            buf_.erase(buf_.begin(), buf_.begin() + 2);
            continue;
        }
        const std::size_t total = 6 + flen;
        if (buf_.size() < total) return;  // 还没收全，等下一次 feed

        // 3) CRC 检查
        const uint16_t crc_rx = static_cast<uint16_t>(buf_[4 + flen]) |
                                (static_cast<uint16_t>(buf_[5 + flen]) << 8);
        if (crc16_ccitt_false(buf_.data() + 2, flen + 2) != crc_rx) {
            ++n_bad_crc_;
            buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(total));
            continue;
        }

        // 4) 交给上层处理
        const int64_t rx_us = host_us();
        ++n_frames_;
        handle(ftype, buf_.data() + 4, flen, rx_us);
        buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(total));
    }
}

void FrameParser::handle(uint8_t type, const uint8_t* payload, std::size_t len, int64_t rx_us) {
    if (type == TYPE_IMU || type == TYPE_IMU_BATCH) {
        for (std::size_t k = 0; k + REC_LEN <= len; k += REC_LEN) {
            const uint8_t* r = payload + k;
            Record rec;
            rec.host_us = rx_us;
            rec.seq     = r[0];
            rec.t_us    = rd_u32(r + 1);
            rec.ax      = rd_i16(r + 5);
            rec.ay      = rd_i16(r + 7);
            rec.az      = rd_i16(r + 9);
            rec.gx      = rd_i16(r + 11);
            rec.gy      = rd_i16(r + 13);
            rec.gz      = rd_i16(r + 15);
            rec.temp    = rd_i16(r + 17);
            records_.push_back(rec);
        }
    } else if (type == TYPE_PONG && len >= 13) {
        const uint8_t id = payload[0];
        int64_t t0 = -1;
        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            if (it->id == id) {
                t0 = it->t0;
                pending_.erase(it);
                break;
            }
        }
        if (t0 < 0) return;  // 不是我们发的（或已经超时清理）
        Pong p;
        p.id = id;
        p.t0 = t0;
        p.t1 = rd_u32(payload + 5);
        p.t2 = rd_u32(payload + 9);
        p.t3 = rx_us;
        pongs_.push_back(p);
    }
}

// ---------------------------------------------------------------- 时钟对齐 --
Sync FrameParser::best_sync() const {
    Sync s;
    if (pongs_.empty()) return s;

    std::vector<double> offs;
    offs.reserve(pongs_.size());
    const Pong* best = nullptr;
    int64_t best_rtt = 0;

    for (const Pong& p : pongs_) {
        // 节点处理时间（注意 uint32 回绕）
        const uint32_t node_proc = static_cast<uint32_t>(p.t2 - p.t1);
        const int64_t  rtt       = (p.t3 - p.t0) - static_cast<int64_t>(node_proc);
        if (best == nullptr || rtt < best_rtt) {
            best     = &p;
            best_rtt = rtt;
        }
        // offset = 节点时钟 - 主机时钟 = ((t1-t0) + (t2-t3)) / 2
        const double off = (static_cast<double>(static_cast<int64_t>(p.t1) - p.t0) +
                            (static_cast<double>(p.t2) - static_cast<double>(p.t3))) / 2.0;
        offs.push_back(off);
    }

    s.valid     = true;
    s.n_pong    = pongs_.size();
    s.rtt_us    = best_rtt;
    s.one_way_us = static_cast<double>(best_rtt) / 2.0;
    s.offset_us = (static_cast<double>(static_cast<int64_t>(best->t1) - best->t0) +
                   (static_cast<double>(best->t2) - static_cast<double>(best->t3))) / 2.0;
    s.jitter_std_us = stddev_of(offs);
    return s;
}

// ------------------------------------------------------------------ 报告 --
void FrameParser::report() const {
    std::printf("\n===== 链路指标 =====\n");
    std::printf("收到字节 %zu，完整帧 %zu，CRC 错 %zu，长度非法 %zu\n",
                n_bytes_, n_frames_, n_bad_crc_, n_bad_len_);

    if (records_.empty()) {
        std::printf("没有解出任何 IMU 记录。排查顺序见 传感器侧加蓝牙方案.md 第 4 节。\n");
        return;
    }

    const std::size_t n = records_.size();
    const double span = static_cast<double>(records_.back().host_us - records_.front().host_us) / 1e6;
    std::printf("记录数 %zu，跨度 %.2f s，平均速率 %.1f Hz（实测）\n",
                n, span, span > 0 ? static_cast<double>(n) / span : 0.0);

    // 按序号统计丢包
    long long lost = 0;
    for (std::size_t i = 1; i < n; ++i) {
        const int d = static_cast<int>(static_cast<uint8_t>(records_[i].seq - records_[i - 1].seq));
        if (d > 1) lost += (d - 1);
    }
    std::printf("序号丢包 %lld / %zu = %.2f%%\n", lost, n, 100.0 * static_cast<double>(lost) /
                                                        static_cast<double>(n > 0 ? n : 1));

    // 采样间隔抖动（用节点自己的时间戳，反映真实采样质量；uint32 回绕用 int32 差处理）
    std::vector<double> dt_ms;
    dt_ms.reserve(n);
    for (std::size_t i = 1; i < n; ++i) {
        const int32_t d = static_cast<int32_t>(records_[i].t_us - records_[i - 1].t_us);
        const double  ms = static_cast<double>(d) / 1000.0;
        if (ms > 0.0 && ms < 100.0) dt_ms.push_back(ms);
    }
    if (!dt_ms.empty()) {
        std::printf("采样间隔(节点时钟) mean=%.3f ms std=%.3f ms p99=%.3f ms\n",
                    mean_of(dt_ms), stddev_of(dt_ms), percentile_of(dt_ms, 99.0));
    }

    // 时钟对齐 + 端到端延迟
    const Sync s = best_sync();
    if (!s.valid) {
        std::printf("\n没收到 PONG（板端→传感器方向没通？），无法做时钟对齐估计。\n");
        return;
    }
    std::printf("\n时钟对齐: RTT=%.3f ms 单程≈%.3f ms offset(节点-主机)=%.1f us (n=%zu)\n",
                static_cast<double>(s.rtt_us) / 1000.0, s.one_way_us / 1000.0,
                s.offset_us, s.n_pong);
    std::printf("偏移估计抖动 std = %.1f us ← 这就是评委问的『同步窗口』\n", s.jitter_std_us);

    // 端到端延迟 = 主机收到时刻 - (节点采样时刻换算到主机时基)
    //   节点时刻 -> 主机时基: t_host = t_us - offset
    std::vector<double> lat_ms;
    lat_ms.reserve(n);
    for (const Record& r : records_) {
        const double ms = (static_cast<double>(r.host_us) -
                           (static_cast<double>(r.t_us) - s.offset_us)) / 1000.0;
        if (ms > -50.0 && ms < 500.0) lat_ms.push_back(ms);
    }
    if (!lat_ms.empty()) {
        std::printf("端到端延迟: mean=%.2f ms p50=%.2f ms p99=%.2f ms max=%.2f ms\n",
                    mean_of(lat_ms), percentile_of(lat_ms, 50.0),
                    percentile_of(lat_ms, 99.0),
                    *std::max_element(lat_ms.begin(), lat_ms.end()));
    }
}

// ------------------------------------------------------------------ CSV ---
std::size_t FrameParser::write_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f) {
        std::printf("[WARN] 打不开 %s，CSV 没写\n", path.c_str());
        return 0;
    }
    f << "host_us,seq,t_us,ax,ay,az,gx,gy,gz,temp\n";
    for (const Record& r : records_) {
        f << r.host_us << ',' << static_cast<int>(r.seq) << ',' << r.t_us << ','
          << r.ax << ',' << r.ay << ',' << r.az << ','
          << r.gx << ',' << r.gy << ',' << r.gz << ',' << r.temp << '\n';
    }
    return records_.size();
}

}  // namespace fitguard
