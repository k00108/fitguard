// =============================================================================
//  bt_imu_uartlite.cpp —— 从 PL 的 axi_uartlite 接收蓝牙 IMU 数据（方案 B 阶段 1）
//  ---------------------------------------------------------------------------
//  为什么用 C++ 而不是 Python：
//    axi_uartlite 的 RX FIFO 只有 16 字节深，115200 下约 1.4 ms 就满。
//    Python 的 MMIO.read 每次进一次 ctypes 调用，轮询速率有限；
//    C++ 直接 mmap /dev/mem，轮询速率高一个数量级以上，基本不会溢出。
//
//  ⚠️ 三个前提：
//    1) 比特流必须先加载好（本程序不加载 bitstream）。
//       最省事的做法：先在 Jupyter 里跑一次 Overlay("design_1.bit")，再来跑这个程序。
//    2) 需要 root（读 /dev/mem）：sudo ./bt_imu_uartlite
//    3) 基地址要和 BD 里 axi_uartlite 的地址一致。你在 add_bt_imu_stage1.tcl
//       里用的是 0x42C00000；如果改过，用 --base 传进来。
//
//  编译（PYNQ 上）：
//     g++ -O2 -std=c++17 -Wall -Wextra bt_imu_proto.cpp bt_imu_uartlite.cpp -o bt_imu_uartlite
//
//  运行：
//     sudo ./bt_imu_uartlite --duration 10
//     sudo ./bt_imu_uartlite --reg              # 只 dump 寄存器，排查用
//     sudo ./bt_imu_uartlite --base 0x42C00000 --duration 30 --csv run1.csv
//
//  axi_uartlite 寄存器（PG142，偏移量 / 字数）
//     0x00 Rx FIFO（读）
//     0x04 Tx FIFO（写）
//     0x08 STAT: bit0 RX有数据 bit1 RX满 bit2 TX空 bit3 TX满
//                bit4 中断 bit5 溢出 bit6 帧错误 bit7 校验错误
//     0x0C CTRL: bit0 复位TX FIFO  bit1 复位RX FIFO  bit2 开中断
// =============================================================================

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "bt_imu_proto.hpp"

using fitguard::FrameParser;

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

// --------------------------------------------------- axi_uartlite 寄存器定义 --
enum : unsigned {
    REG_RX   = 0x00 >> 2,
    REG_TX   = 0x04 >> 2,
    REG_STAT = 0x08 >> 2,
    REG_CTRL = 0x0C >> 2,
};
constexpr uint32_t STAT_RX_VALID = 1u << 0;
constexpr uint32_t STAT_TX_EMPTY = 1u << 2;
constexpr uint32_t STAT_TX_FULL  = 1u << 3;
constexpr uint32_t STAT_OVERRUN  = 1u << 5;
constexpr uint32_t STAT_FRAME_ERR= 1u << 6;
constexpr uint32_t CTRL_RST_TX   = 1u << 0;
constexpr uint32_t CTRL_RST_RX   = 1u << 1;

class AxiUartLite {
public:
    static constexpr std::size_t kMapSize = 0x1000;   // 4 KB 足够覆盖 4 个寄存器

    bool open(uint32_t base) {
        fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
        if (fd_ < 0) {
            std::fprintf(stderr, "打不开 /dev/mem: %s（是不是忘了 sudo？）\n", std::strerror(errno));
            return false;
        }
        // mmap 的 offset 必须是页对齐的；0x42C00000 是 4K 对齐，正好可以直接用
        void* p = ::mmap(nullptr, kMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                         static_cast<off_t>(base));
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "mmap 0x%08X 失败: %s\n", base, std::strerror(errno));
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        regs_ = static_cast<volatile uint32_t*>(p);
        base_ = base;
        return true;
    }

    ~AxiUartLite() {
        if (regs_) ::munmap(const_cast<uint32_t*>(regs_), kMapSize);
        if (fd_ >= 0) ::close(fd_);
    }

    uint32_t base() const { return base_; }
    uint32_t stat() const { return regs_[REG_STAT]; }

    void reset_fifos() { regs_[REG_CTRL] = CTRL_RST_RX | CTRL_RST_TX; }

    // 把 RX FIFO 里的字节全读出来，返回读到的个数
    std::size_t drain(uint8_t* out, std::size_t cap) {
        std::size_t n = 0;
        while (n < cap && (stat() & STAT_RX_VALID)) {
            out[n++] = static_cast<uint8_t>(regs_[REG_RX] & 0xFF);
        }
        return n;
    }

    // 把一帧写进 TX FIFO，然后等 FIFO 空（表示最后一个字节已经进了移位寄存器）
    bool send(const uint8_t* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            while (stat() & STAT_TX_FULL) { /* 等空位 */ }
            regs_[REG_TX] = p[i];
        }
        while (!(stat() & STAT_TX_EMPTY)) { /* 等发完 */ }
        return true;
    }

    void dump() const {
        const uint32_t st = stat();
        std::printf("axi_uartlite @ 0x%08X\n", base_);
        std::printf("  STAT = 0x%08X  RX有数据=%d RX满=%d TX空=%d TX满=%d "
                    "溢出=%d 帧错误=%d\n",
                    st,
                    (st & STAT_RX_VALID) ? 1 : 0, (st & (1u << 1)) ? 1 : 0,
                    (st & STAT_TX_EMPTY) ? 1 : 0, (st & STAT_TX_FULL) ? 1 : 0,
                    (st & STAT_OVERRUN) ? 1 : 0, (st & STAT_FRAME_ERR) ? 1 : 0);
    }

private:
    int                       fd_   = -1;
    volatile uint32_t*        regs_ = nullptr;
    uint32_t                  base_ = 0;
};

struct Options {
    uint32_t    base          = 0x42C00000u;
    double      duration_s    = 10.0;
    std::string csv           = "bt_imu_uartlite_log.csv";
    double      ping_interval = 0.5;
    bool        reg_only      = false;
};

void parse_args(int argc, char** argv, Options& o) {
    auto need_value = [&](int& i, const std::string& key) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "参数 %s 后面缺值\n", key.c_str());
            std::exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i], key = a, val;
        const std::size_t eq = a.find('=');
        if (eq != std::string::npos) { key = a.substr(0, eq); val = a.substr(eq + 1); }
        auto value = [&]() { return val.empty() ? need_value(i, key) : val; };

        if (key == "--base")               o.base = static_cast<uint32_t>(std::strtoul(value().c_str(), nullptr, 0));
        else if (key == "--duration")      o.duration_s = std::atof(value().c_str());
        else if (key == "--csv")           o.csv = value();
        else if (key == "--ping-interval") o.ping_interval = std::atof(value().c_str());
        else if (key == "--reg")           o.reg_only = true;
        else if (key == "-h" || key == "--help") {
            std::printf(
                "用法: sudo bt_imu_uartlite [选项]\n"
                "  --base <addr>          axi_uartlite 基地址（默认 0x42C00000）\n"
                "  --duration <s>         接收秒数（默认 10）\n"
                "  --csv <path>           CSV 输出（默认 bt_imu_uartlite_log.csv）\n"
                "  --ping-interval <s>    发 PING 的间隔，0 关闭（默认 0.5）\n"
                "  --reg                  只 dump 寄存器后退出（排查用）\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "未知参数: %s（用 --help 看用法）\n", key.c_str());
            std::exit(2);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    parse_args(argc, argv, opt);

    AxiUartLite uart;
    if (!uart.open(opt.base)) return 1;

    uart.reset_fifos();
    if (opt.reg_only) {
        ::usleep(100 * 1000);
        uart.dump();
        return 0;
    }
    std::printf("axi_uartlite @ 0x%08X 就绪，开始接收 %.0f 秒...（Ctrl-C 可提前结束）\n",
                opt.base, opt.duration_s);
    std::printf("提示：如果收到了字节但全是垃圾，先检查比特流里 C_S_AXI_ACLK_FREQ_HZ "
                "是否等于真实 AXI 时钟（你们是 50 MHz，IP 默认 100 MHz 会让波特率减半）\n");

    std::signal(SIGINT, on_sigint);

    FrameParser parser;
    uint8_t     rbuf[256];

    const int64_t t_start      = fitguard::host_us();
    int64_t       t_last_ping  = 0;
    int64_t       t_last_print = t_start;
    long long     n_polls      = 0;
    bool          ping_ok      = opt.ping_interval > 0.0;
    bool          last_overrun = false, last_ferr = false;

    while (!g_stop) {
        const int64_t now = fitguard::host_us();
        if (static_cast<double>(now - t_start) / 1e6 >= opt.duration_s) break;

        // ---- 轮询抽干 RX FIFO ----
        ++n_polls;
        const std::size_t n = uart.drain(rbuf, sizeof(rbuf));
        if (n > 0) parser.feed(rbuf, n);

        // ---- 错误标志变化时告警（演示时很有说服力）----
        const uint32_t st = uart.stat();
        const bool ov = (st & STAT_OVERRUN) != 0, fe = (st & STAT_FRAME_ERR) != 0;
        if (ov != last_overrun) {
            std::printf("[UART] OVERRUN -> %s（FIFO 溢出，轮询没跟上）\n", ov ? "置位" : "清零");
            last_overrun = ov;
        }
        if (fe != last_ferr) {
            std::printf("[UART] FRAME_ERROR -> %s（波特率/接线问题）\n", fe ? "置位" : "清零");
            last_ferr = fe;
        }

        // ---- 周期性发 PING ----
        if (ping_ok && static_cast<double>(now - t_last_ping) / 1e6 >= opt.ping_interval) {
            t_last_ping = now;
            uint8_t id = 0;
            const std::vector<uint8_t> ping = parser.make_ping(&id);
            uart.send(ping.data(), ping.size());
            // 等 TX FIFO 空之后再打时间戳，t0 更接近"真正上线的时刻"
            parser.restamp_ping(id, fitguard::host_us());
        }

        // ---- 每秒打印一次 ----
        if (static_cast<double>(now - t_last_print) / 1e6 >= 1.0) {
            const double el = static_cast<double>(now - t_start) / 1e6;
            std::printf("[%5.1fs] bytes=%8zu frames=%6zu recs=%6zu badcrc=%zu badlen=%zu "
                        "polls/s=%.0f\n",
                        el, parser.n_bytes(), parser.n_frames(), parser.records().size(),
                        parser.n_bad_crc(), parser.n_bad_len(),
                        static_cast<double>(n_polls) / (el > 0 ? el : 1.0));
            t_last_print = now;
        }
    }

    parser.report();
    if (!opt.csv.empty()) {
        const std::size_t rows = parser.write_csv(opt.csv);
        std::printf("已写入 %s（%zu 行）\n", opt.csv.c_str(), rows);
    }
    std::printf("\n轮询总次数 %lld，平均 %.0f 次/秒（C++ 直读 /dev/mem）\n",
                n_polls, static_cast<double>(n_polls) /
                         ((static_cast<double>(fitguard::host_us() - t_start) / 1e6) + 1e-9));
    return 0;
}
