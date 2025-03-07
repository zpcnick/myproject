import numpy as np
from PyMieScatt import MieQ
from meep.materials import Ag

# 定义波长范围（单位：纳米）
wavelengths_nm = np.linspace(250, 750, 200)  # 200 samples
# 定义半径列表（单位：纳米）
#r_values = [6, 7, 8, 9, 10]
r_values = [20, 30, 40, 50, 100]

print(Ag.valid_freq_range)

for r_current in r_values:
    # 生成文件名
    filename = f"r{r_current}.dat"
    # 打开文件以写入数据
    with open(filename, "w") as f:
        # 写入表头
        f.write("# Wavelength(nm) Absorption(nm²) Scattering(nm²) Extinction(nm²)\n")
        # 遍历波长范围
        for wavelength_nm in wavelengths_nm:
            # 计算频率（单位：微米^-1）
            freq = 1000 / wavelength_nm
            # 获取银的介电常数并计算平均折射率
            epsilon = np.trace(Ag.epsilon(freq)) / 3
            n = np.sqrt(epsilon.real + 1j * epsilon.imag)
            
            # 计算米氏散射参数（直径单位：纳米）
            mie_result = MieQ(
                n, 
                wavelength_nm, 
                2 * r_current,  # 直径 = 2r
                asCrossSection=True,
                asDict=True
            )
            
            # 写入数据
            print(wavelength_nm,mie_result['Cabs'],mie_result['Csca'],mie_result['Cext'], file=f)

print("data saved in r.dat files")

