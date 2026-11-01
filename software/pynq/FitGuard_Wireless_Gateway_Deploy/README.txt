FitGuard 无线网关增量包

1. 本目录独立于原 FitGuard_Unified_Test，不覆盖原 Notebook、模型、bit/hwh 或数据集。
2. 先运行 FitGuard_Wireless_Check.ipynb，确认串口、CRC、50 Hz 和丢包率。
3. 检查通过后运行 FitGuard_Wireless_Live_Demo.ipynb。
4. 两个 Notebook 通过 ../FitGuard_Unified_Test 复用已经验证过的视觉、模型和 bitstream 文件。
5. 运行期间不要同时打开串口监视器、VOFA 或终端 od/cat，否则 /dev/ttyUSB0 会被抢占。
