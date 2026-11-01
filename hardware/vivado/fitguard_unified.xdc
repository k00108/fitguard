## FitGuard MPU6050 I2C on PYNQ-Z2 PMODA
## Reused from the already validated MPU6050 project.
set_property PACKAGE_PIN Y18 [get_ports MPU6050_IIC_scl_io]
set_property IOSTANDARD LVCMOS33 [get_ports MPU6050_IIC_scl_io]

set_property PACKAGE_PIN Y19 [get_ports MPU6050_IIC_sda_io]
set_property IOSTANDARD LVCMOS33 [get_ports MPU6050_IIC_sda_io]

