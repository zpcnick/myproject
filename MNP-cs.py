import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import meep.materials as mat

# 参数设置
r = 0.01         # 银球半径 (μm)
wvl_min, wvl_max = 0.2, 0.7   # 波长范围 (μm)
frq_min = 1 / wvl_max         # 最小频率 (1/μm)
frq_max = 1 / wvl_min         # 最大频率 (1/μm)
frq_cen = (frq_min + frq_max)/2  # 中心频率
dfrq = frq_max - frq_min      # 频率带宽
nfrq = 200                    # 频率采样数
resol = 1000                  # 分辨率（每微米像素数）
dpml = 0.02                   # PML厚度
s = 2*(dpml + 0.03 + r)            # 模拟区域尺寸

geometry = [mp.Sphere(radius=r, material=mat.Ag, center=mp.Vector3())]
sources = [mp.Source(
    mp.GaussianSource(frq_cen, fwidth=dfrq*2, is_integrated=True),
    center=mp.Vector3(-s/2 + dpml, 0, 0),
    size=mp.Vector3(0, s, s),
    component=mp.Ez
)]
cell_size = mp.Vector3(s, s, s)
pml_layers = [mp.PML(dpml)]
bx = 2*r
by = 2*r
bz = 2*r
f_peaks = [3.2, ]      # unit eV, for dft_fields

# 通量监视器设置
box_regions = [
    mp.FluxRegion(center=mp.Vector3(x=-bx/2), size=mp.Vector3(0, by, bz)),
    mp.FluxRegion(center=mp.Vector3(x=+bx/2), size=mp.Vector3(0, by, bz)),
    mp.FluxRegion(center=mp.Vector3(y=-by/2), size=mp.Vector3(bx, 0, bz)),
    mp.FluxRegion(center=mp.Vector3(y=+by/2), size=mp.Vector3(bx, 0, bz)),
    mp.FluxRegion(center=mp.Vector3(z=-bz/2), size=mp.Vector3(bx, by, 0)),
    mp.FluxRegion(center=mp.Vector3(z=+bz/2), size=mp.Vector3(bx, by, 0)),
]

# 空模拟（无银球）
sim_empty = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    resolution=resol
)

### settings for sim_empty.add, flux and dft
objs_empty = [sim_empty.add_flux(frq_cen, dfrq, nfrq, region) for region in box_regions]
xz0plane_pml = mp.Volume(center=mp.Vector3(), size=mp.Vector3(s - 2 * dpml, 0, s - 2 * dpml))
dft_source = sim_empty.add_dft_fields([mp.Ez], np.array(f_peaks)/1.24, where=xz0plane_pml)

sim_empty.run(until_after_sources=10)

### after run, sim_empty.get
data_empty = [sim_empty.get_flux_data(obj) for obj in objs_empty]
freqs = np.array(mp.get_flux_freqs(objs_empty[0]))
inc = np.array(mp.get_fluxes(objs_empty[0])) / (by*bz)

ez_data_source = sim_empty.get_dft_array(dft_source, mp.Ez, 0)
ez_intensity_source = np.abs(ez_data_source)

# 含银球模拟
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    geometry=geometry,
    resolution=resol
)

### settings for sim.add, flux and dft
objs_abs = [sim.add_flux(frq_cen, dfrq, nfrq, region) for region in box_regions]
objs_scat = [sim.add_flux(frq_cen, dfrq, nfrq, region) for region in box_regions]
for obj, data in zip(objs_scat, data_empty):
    sim.load_minus_flux_data(obj, data)
dft_fields = sim.add_dft_fields([mp.Ez], np.array(f_peaks)/1.24, where=xz0plane_pml)

sim.run(until_after_sources=30)

### after run, sim.get
data_abs = [np.array(mp.get_fluxes(obj)) for obj in objs_abs]
data_scat = [np.array(mp.get_fluxes(obj)) for obj in objs_scat]
flux_abs = (
      data_abs[0] - data_abs[1]  # x方向净通量
    + data_abs[2] - data_abs[3]  # y方向净通量
    + data_abs[4] - data_abs[5]  # z方向净通量
)
flux_scat = (
      data_scat[0] - data_scat[1]  # x方向净通量
    + data_scat[2] - data_scat[3]  # y方向净通量
    + data_scat[4] - data_scat[5]  # z方向净通量
)

# 计算截面 (单位：nm²)
sigma_abs = flux_abs / inc * 1e6
sigma_scat = - flux_scat / inc *1e6
sigma_ext = sigma_abs + sigma_scat

# 获取并保存电场分布
ez_data = sim.get_dft_array(dft_fields, mp.Ez, 0)
ez_intensity = np.abs(ez_data)
ez_enhancement = ez_intensity / ez_intensity_source

# 保存数据
np.savetxt('spectra.dat', np.column_stack((freqs*1.24, sigma_abs, sigma_scat)),
           header='freqs(eV)\tAbsorption\tScattering')
np.savetxt('ez2.dat', ez_enhancement)

# 绘制光谱
if mp.am_master():
    plt.figure(dpi=200)
    plt.plot(freqs*1.24, sigma_ext, 'k-', label='Extinction')
    plt.plot(freqs*1.24, sigma_scat, 'r--', label='Scattering')
    plt.plot(freqs*1.24, sigma_abs, 'b:', label='Absorption')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Cross-section (nm²)')
    plt.xlim((frq_min*1.24,frq_max*1.24))
    plt.legend()
    plt.savefig('spectra.png')
    plt.close()

    plt.figure(dpi=200)
    plt.imshow(ez_enhancement.T, cmap='RdPu', interpolation='spline36')
    plt.colorbar(label="Electric Field Enhancement")
    plt.xlabel("x (points)")
    plt.ylabel("z (points)")
    plt.title(f"Electric Field Enhancement Distribution @ {f_peaks[0]} eV")
    plt.savefig("ez2.png")
    plt.close()

