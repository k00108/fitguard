// =============================================================================
//  bt_imu_serial.cpp —— 从【USB 串口】接收蓝牙 IMU 数据（方案 A）
//  ---------------------------------------------------------------------------
//  适用两种场景，同一个程序：
//    1) PYNQ 上：第二块 ESP32 当"蓝牙→USB串口"桥，插在 PYNQ 的 USB 口
//                设备名一般是 /dev/ttyUSB0（CP2102/CH340）或 /dev/ttyACM0（原生USB）
//    2) 开发机上：直接拿一个 USB-TTL 接桥模块的 TX/RX，先在电脑上把链路调通
//                （Windows 请用 WSL；本程序是 POSIX/Linux 的）
//
//  为什么用 C++ 而不是 Python：
//    · 轮询串口/寄存器不再受解释器调度影响，抖动和丢字节概率都更低
//    · 不需要 pyserial / numpy，PYNQ 上只要有 g++ 就能编
//
//  编译（PYNQ / Ubuntu / WSL）：
//     g++ -O2 -std=c++17 -Wall -Wextra bt_imu_proto.cpp bt_imu_serial.cpp -o bt_imu_serial
//
//  运行：
//     sudo ./bt_imu_serial --port /dev/ttyUSB0 --baud 921600 --duration 10
//     ./bt_imu_serial --list                      # 只列出串口设备
//     sudo ./bt_imu_serial --port /dev/rfcomm0 --duration 30   # 若走 Linux 蓝牙 RFCOMM
// =============================================================================

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1        // 让 glibc 暴露 B921600 这类扩展波特率宏 / glob 等
#endif

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <glob.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include "bt_imu_proto.hpp"

using fitguard::FrameParser;

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

struct Options {
    std::string port          = "/dev/ttyUSB0";
    int         baud          = 921600;
    double      duration_s    = 10.0;
    std::string csv           = "bt_imu_serial_log.csv";
    double      ping_interval = 0.5;   // 秒；0 = 不发 PING
    bool        list_only     = false;
};

// 支持 "--key value" 和 "--key=value" 两种写法
void parse_args(int argc, char** argv, Options& o) {
    auto need_value = [&](int& i, const std::string& key) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "参数 %s 后面缺值\n", key.c_str());
            std::exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string key = a, val;
        const std::size_t eq = a.find('=');
        if (eq != std::string::npos) {
            key = a.substr(0, eq);
            val = a.substr(eq + 1);
        }
        auto value = [&]() { return val.empty() ? need_value(i, key) : val; };

        if (key == "--port")               o.port = value();
        else if (key == "--baud")          o.baud = std::atoi(value().c_str());
        else if (key == "--duration")      o.duration_s = std::atof(value().c_str());
        else if (key == "--csv")           o.csv = value();
        else if (key == "--ping-interval") o.ping_interval = std::atof(value().c_str());
        else if (key == "--list")          o.list_only = true;
        else if (key == "-h" || key == "--help") {
            std::printf(
                "用法: bt_imu_serial [选项]\n"
                "  --port <dev>          串口设备（默认 /dev/ttyUSB0）\n"
                "  --baud <n>            波特率（默认 921600）\n"
                "  --duration <s>        接收秒数（默认 10）\n"
                "  --csv <path>          CSV 输出（默认 bt_imu_serial_log.csv）\n"
                "  --ping-interval <s>   发 PING 的间隔，0 关闭（默认 0.5）\n"
                "  --list                只列出串口设备\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "未知参数: %s（用 --help 看用法）\n", key.c_str());
            std::exit(2);
        }
    }
}

void list_serial_ports() {
    const char* patterns[] = {"/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyAMA*",
                              "/dev/ttyPS*",  "/dev/rfcomm*", "/dev/serial/by-id/*"};
    bool found = false;
    for (const char* pat : patterns) {
        glob_t g{};   // 必须清零：没匹配时 globfree 作用在已初始化的结构体上才安全
        if (glob(pat, 0, nullptr, &g) == 0) {
            for (std::size_t i = 0; i < g.gl_pathc; ++i) {
                std::printf("  %s\n", g.gl_pathv[i]);
                found = true;
            }
        }
        globfree(&g);
    }
    if (!found) {
        std::printf("没有发现任何串口设备。检查：USB 线是否插好、桥模块是否上电、\n"
                    "dmesg | tail 看有没有认到 cp210x/ch341/cdc_acm\n");
    }
}

bool apply_baud(termios& tio, int baud) {
    speed_t s;
    switch (baud) {
        case 9600:   s = B9600;   break;
        case 19200:  s = B19200;  break;
        case 38400:  s = B38400;  break;
        case 57600:  s = B57600;  break;
        case 115200: s = B115200; break;
        case 230400: s = B230400; break;
#ifdef B460800
        case 460800: s = B460800; break;
#endif
#ifdef B921600
        case 921600: s = B921600; break;
#endif
#ifdef B1000000
        case 1000000: s = B1000000; break;
#endif
        default:
            std::fprintf(stderr, "不支持这个波特率: %d\n", baud);
            return false;
    }
    cfsetospeed(&tio, s);
    cfsetispeed(&tio, s);
    return true;
}

// 手动实现 raw 模式（不用 cfmakeraw，避免 -std=c++17 下的扩展宏问题）
void set_raw(termios& t) {
    t.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    t.c_oflag &= ~OPOST;
    t.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    t.c_cflag &= ~(CSIZE | PARENB | CSTOPB);
    t.c_cflag |= (CS8 | CLOCAL | CREAD);
#ifdef CRTSCTS
    t.c_cflag &= ~CRTSCTS;        // 关硬件流控
#endif
    t.c_cc[VMIN]  = 0;
    t.c_cc[VTIME] = 0;
}

int open_serial(const std::string& port, int baud) {
    const int fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        std::fprintf(stderr, "打不开 %s: %s\n", port.c_str(), std::strerror(errno));
        std::fprintf(stderr, "提示：PYNQ 上默认用户是 xilinx，通常要 sudo，或者 "
                             "sudo chmod 666 %s\n", port.c_str());
        return -1;
    }
    termios tio{};
    if (tcgetattr(fd, &tio) != 0) {
        std::fprintf(stderr, "tcgetattr 失败: %s\n", std::strerror(errno));
        ::close(fd);
        return -1;
    }
    if (!apply_baud(tio, baud)) {
        ::close(fd);
        return -1;
    }
    set_raw(tio);
    if (tcsetattr(fd, TCSANOW, &tio) != 0) {
        std::fprintf(stderr, "tcsetattr 失败: %s\n", std::strerror(errno));
        ::close(fd);
        return -1;
    }
    tcflush(fd, TCIOFLUSH);
    return fd;
}

bool write_all(int fd, const uint8_t* p, std::size_t n) {
    std::size_t sent = 0;
    while (sent < n) {
        const ssize_t w = ::write(fd, p + sent, n - sent);
        if (w > 0) {
            sent += static_cast<std::size_t>(w);
        } else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd pfd{fd, POLLOUT, 0};
            poll(&pfd, 1, 5);
        } else {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    parse_args(argc, argv, opt);

    if (opt.list_only) {
        std::printf("可用串口设备：\n");
        list_serial_ports();
        return 0;
    }

    std::signal(SIGINT, on_sigint);

    const int fd = open_serial(opt.port, opt.baud);
    if (fd < 0) {
        list_serial_ports();
        return 1;
    }
    std::printf("已打开 %s @ %d，开始接收 %.0f 秒...（Ctrl-C 可提前结束）\n",
                opt.port.c_str(), opt.baud, opt.duration_s);

    FrameParser parser;
    uint8_t     rbuf[4096];

    const int64_t t_start      = fitguard::host_us();
    int64_t       t_last_ping  = 0;
    int64_t       t_last_print = t_start;
    bool          ping_ok      = opt.ping_interval > 0.0;

    while (!g_stop) {
        const int64_t now = fitguard::host_us();
        if (static_cast<double>(now - t_start) / 1e6 >= opt.duration_s) break;

        // ---- 读数据：一次尽量抽干内核缓冲 ----
        pollfd pfd{fd, POLLIN, 0};
        poll(&pfd, 1, 2);   // 2 ms 超时，顺便让出 CPU 给 PING 调度和 Ctrl-C
        if (pfd.revents & POLLIN) {
            for (;;) {
                const ssize_t n = ::read(fd, rbuf, sizeof(rbuf));
                if (n > 0) {
                    parser.feed(rbuf, static_cast<std::size_t>(n));
                } else {
                    break;   // EAGAIN / 无数据
                }
            }
        }

        // ---- 周期性发 PING（做时钟对齐）----
        if (ping_ok && static_cast<double>(now - t_last_ping) / 1e6 >= opt.ping_interval) {
            t_last_ping = now;
            const std::vector<uint8_t> ping = parser.make_ping();
            if (!write_all(fd, ping.data(), ping.size())) {
                std::printf("[WARN] 发 PING 失败（只收不发也能用），关闭 PING\n");
                ping_ok = false;
            }
        }

        // ---- 每秒打印一次链路状态 ----
        if (static_cast<double>(now - t_last_print) / 1e6 >= 1.0) {
            t_last_print = now;
            std::printf("[%5.1fs] bytes=%8zu frames=%6zu recs=%6zu badcrc=%zu badlen=%zu pongs=%zu\n",
                        static_cast<double>(now - t_start) / 1e6,
                        parser.n_bytes(), parser.n_frames(), parser.records().size(),
                        parser.n_bad_crc(), parser.n_bad_len(), parser.pongs().size());
        }
    }

    ::close(fd);

    parser.report();
    if (!opt.csv.empty()) {
        const std::size_t rows = parser.write_csv(opt.csv);
        std::printf("已写入 %s（%zu 行）—— 画图可以用 Jupyter / gnuplot / Excel\n",
                    opt.csv.c_str(), rows);
    }
    return 0;
}
