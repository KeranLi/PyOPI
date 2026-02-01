#!/usr/bin/env python
"""调试降水计算中的问题"""

import sys
sys.path.insert(0, 'OPI_python')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid

from opi.physics.fourier import fourier_solution
from opi.physics.precipitation import isotherm
from opi.physics.thermodynamics import base_state

# 参数设置
x = np.linspace(-200e3, 200e3, 100)
y = np.linspace(-100e3, 100e3, 50)
X, Y = np.meshgrid(x, y)

# 高斯山脉
mountain_height = 3000
sigma_x = 30e3
sigma_y = 50e3
h_grid = mountain_height * np.exp(-(X**2)/(2*sigma_x**2) - (Y**2)/(2*sigma_y**2))

U = 10.0
azimuth = 90.0
T0 = 290.0
M = 0.25
kappa = 0.0
tau_c = 1000.0
f_c = 1e-4
f_p0 = 1.0

h_max = h_grid.max()
NM = M * U / h_max

print("="*60)
print("Debug OPI Precipitation Calculation")
print("="*60)

# 基础状态
z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
print(f"\nBase State:")
print(f"  rho_s0 = {rho_s0:.6f} kg/m^3")
print(f"  h_s = {h_s:.1f} m")
print(f"  h_rho = {h_rho:.1f} m")
print(f"  gamma_ratio = {gamma_ratio:.4f}")

# Fourier解
fourier_result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)

s = fourier_result['s']
t = fourier_result['t']
Sxy = fourier_result['Sxy']
Txy = fourier_result['Txy']
h_wind = fourier_result['h_wind']
k_s = fourier_result['k_s']
k_t = fourier_result['k_t']
h_hat = fourier_result['h_hat']
k_z = fourier_result['k_z']

print(f"\nFourier Solution:")
print(f"  s shape: {s.shape}, range: [{s.min():.0f}, {s.max():.0f}]")
print(f"  t shape: {t.shape}, range: [{t.min():.0f}, {t.max():.0f}]")
print(f"  h_wind shape: {h_wind.shape}")
print(f"  h_hat shape: {h_hat.shape}")
print(f"  k_z shape: {k_z.shape}")

# 网格参数
d_s = s[1] - s[0]
n_s_grid, n_t_grid = h_wind.shape
n_s_pad, n_t_pad = h_hat.shape

print(f"\nGrid Parameters:")
print(f"  d_s = {d_s:.1f} m")
print(f"  n_s_grid = {n_s_grid}, n_t_grid = {n_t_grid}")
print(f"  n_s_pad = {n_s_pad}, n_t_pad = {n_t_pad}")

# 计算258K等温线高度
z258_wind = isotherm(258.0, z_bar, T, gamma_env, gamma_sat, h_rho, 
                     n_s_grid, n_t_grid, h_hat, k_z)

# 下落时间
w_f_snow = -1.0
w_f_rain = -6.0

tau_f_grid = np.where(
    z258_wind <= h_wind,
    -h_s / w_f_snow,
    -(z258_wind - h_wind) / w_f_rain - h_s * np.exp(-(z258_wind - h_wind) / h_s) / w_f_snow
)
tau_f = np.mean(tau_f_grid)

print(f"\nFall Time:")
print(f"  tau_f = {tau_f:.1f} s")
print(f"  tau_f_grid range: [{tau_f_grid.min():.1f}, {tau_f_grid.max():.1f}]")

# Green's functions
k_s_col = k_s.reshape(-1, 1)

GS_hat = (gamma_ratio * rho_s0 * 1j * k_s_col * U / 
          (1 - h_s * (1j * k_z + 1/(2*h_rho))))

# FIXME: Temporary fix for precipitation scaling issue
scale_factor = 1e-4
GS_hat = GS_hat * scale_factor

GC_hat = 1.0 / (tau_c * (kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 1)
GF_hat = 1.0 / (tau_f * (kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 1)

print(f"\nGreen's Functions:")
print(f"  GS_hat max abs: {np.abs(GS_hat).max():.6e}")
print(f"  GC_hat max abs: {np.abs(GC_hat).max():.6f}")
print(f"  GF_hat max abs: {np.abs(GF_hat).max():.6f}")

# 参考降水率
p_star_hat = GS_hat * GC_hat * GF_hat * h_hat
print(f"  p_star_hat max abs: {np.abs(p_star_hat).max():.6e}")

p_star_pos_wind = np.fft.ifft2(p_star_hat)
p_star_pos_wind = np.real(p_star_pos_wind)
p_star_pos_wind = p_star_pos_wind[:n_s_grid, :n_t_grid]
p_star_pos_wind[p_star_pos_wind < 0] = 0

print(f"\nReference Precipitation (p_star_pos_wind):")
print(f"  Shape: {p_star_pos_wind.shape}")
print(f"  Min: {p_star_pos_wind.min():.6e} kg/m^2/s")
print(f"  Max: {p_star_pos_wind.max():.6e} kg/m^2/s")
print(f"  Mean: {p_star_pos_wind.mean():.6e} kg/m^2/s")
print(f"  Max (mm/day): {p_star_pos_wind.max()*1000*86400:.2f} mm/day")

# 云水和降水柱密度
QC_star_pos_wind = np.fft.ifft2(tau_c * GS_hat * GC_hat * h_hat)
QC_star_pos_wind = np.real(QC_star_pos_wind)
QC_star_pos_wind = QC_star_pos_wind[:n_s_grid, :n_t_grid]
QC_star_pos_wind[QC_star_pos_wind < 0] = 0

QF_star_pos_wind = np.fft.ifft2(tau_f * GS_hat * GC_hat * GF_hat * h_hat)
QF_star_pos_wind = np.real(QF_star_pos_wind)
QF_star_pos_wind = QF_star_pos_wind[:n_s_grid, :n_t_grid]
QF_star_pos_wind[QF_star_pos_wind < 0] = 0

QT_star_pos_wind = rho_s0 * h_s + QC_star_pos_wind + QF_star_pos_wind

print(f"\nColumn Densities:")
print(f"  rho_s0*h_s (vapor): {rho_s0*h_s:.4f} kg/m^2")
print(f"  QC_star max: {QC_star_pos_wind.max():.4f} kg/m^2")
print(f"  QF_star max: {QF_star_pos_wind.max():.4f} kg/m^2")
print(f"  QT_star max: {QT_star_pos_wind.max():.4f} kg/m^2")
print(f"  QT_star min: {QT_star_pos_wind.min():.4f} kg/m^2")

# 蒸发循环
if f_p0 == 1.0:
    r_h_wind = np.ones((n_s_grid, n_t_grid))
    f_p_wind = np.ones((n_s_grid, n_t_grid))
else:
    # 计算代码...
    pass

print(f"\nEvaporation Recycling (f_p0={f_p0}):")
print(f"  f_p_wind: all ones = {np.all(f_p_wind == 1)}")

# 水汽比例积分
integrand = f_p_wind * p_star_pos_wind / QT_star_pos_wind
print(f"\nIntegrand for moisture correction:")
print(f"  Shape: {integrand.shape}")
print(f"  Min: {integrand.min():.6e}")
print(f"  Max: {integrand.max():.6e}")
print(f"  Mean: {integrand.mean():.6e}")

# 使用cumulative_trapezoid积分
integral = cumulative_trapezoid(integrand, dx=d_s, axis=0, initial=0)
print(f"\nIntegral result:")
print(f"  Shape: {integral.shape}")
print(f"  Min: {integral.min():.6e}")
print(f"  Max: {integral.max():.6e}")

# 计算水汽比例
f_v_wind = (rho_s0 * h_s / QT_star_pos_wind) * np.exp(-(1.0/U) * integral)
print(f"\nVapor Ratio (f_v_wind):")
print(f"  Shape: {f_v_wind.shape}")
print(f"  Min: {f_v_wind.min():.6f}")
print(f"  Max: {f_v_wind.max():.6f}")
print(f"  Mean: {f_v_wind.mean():.6f}")

# 计算最终降水
p_wind = f_v_wind * p_star_pos_wind
print(f"\nFinal Precipitation in Wind Coordinates (p_wind):")
print(f"  Shape: {p_wind.shape}")
print(f"  Min: {p_wind.min():.6e} kg/m^2/s")
print(f"  Max: {p_wind.max():.6e} kg/m^2/s")
print(f"  Mean: {p_wind.mean():.6e} kg/m^2/s")
print(f"  Max (mm/day): {p_wind.max()*1000*86400:.2f} mm/day")

# 插值回地理网格
interp_func = RegularGridInterpolator(
    (s, t), p_wind,
    method='linear',
    bounds_error=False,
    fill_value=0
)

# 检查查询点形状
query_points = np.column_stack([Sxy.ravel(), Txy.ravel()])
print(f"\nInterpolation:")
print(f"  Query points shape: {query_points.shape}")
print(f"  Sxy range: [{Sxy.min():.0f}, {Sxy.max():.0f}]")
print(f"  Txy range: [{Txy.min():.0f}, {Txy.max():.0f}]")
print(f"  s range: [{s.min():.0f}, {s.max():.0f}]")
print(f"  t range: [{t.min():.0f}, {t.max():.0f}]")

p_grid_flat = interp_func(query_points)
p_grid = p_grid_flat.reshape(Sxy.shape)

print(f"\nFinal Precipitation in Geographic Coordinates (p_grid):")
print(f"  Shape: {p_grid.shape}")
print(f"  Min: {p_grid.min():.6e} kg/m^2/s")
print(f"  Max: {p_grid.max():.6e} kg/m^2/s")
print(f"  Mean: {p_grid.mean():.6e} kg/m^2/s")
print(f"  Max (mm/day): {p_grid.max()*1000*86400:.2f} mm/day")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 地形
ax = axes[0, 0]
im = ax.pcolormesh(X/1000, Y/1000, h_grid, shading='auto', cmap='terrain')
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('Topography (m)')
plt.colorbar(im, ax=ax)
ax.annotate('', xy=(150, -80), xytext=(100, -80),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# p_star (参考降水)
ax = axes[0, 1]
# 创建s,t网格用于绘图
S_wind, T_wind = np.meshgrid(s, t, indexing='ij')
im = ax.pcolormesh(S_wind/1000, T_wind/1000, p_star_pos_wind*1000*86400, 
                   shading='auto', cmap='Blues')
ax.set_xlabel('s (km)')
ax.set_ylabel('t (km)')
ax.set_title('p_star (mm/day) - Wind Coords')
plt.colorbar(im, ax=ax)

# p_wind (最终降水，风坐标)
ax = axes[0, 2]
im = ax.pcolormesh(S_wind/1000, T_wind/1000, p_wind*1000*86400, 
                   shading='auto', cmap='Blues')
ax.set_xlabel('s (km)')
ax.set_ylabel('t (km)')
ax.set_title('p_wind (mm/day) - Wind Coords')
plt.colorbar(im, ax=ax)

# f_v (水汽比例)
ax = axes[1, 0]
im = ax.pcolormesh(S_wind/1000, T_wind/1000, f_v_wind, 
                   shading='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xlabel('s (km)')
ax.set_ylabel('t (km)')
ax.set_title('Vapor Ratio f_v')
plt.colorbar(im, ax=ax)

# QT (总水汽)
ax = axes[1, 1]
im = ax.pcolormesh(S_wind/1000, T_wind/1000, QT_star_pos_wind, 
                   shading='auto', cmap='viridis')
ax.set_xlabel('s (km)')
ax.set_ylabel('t (km)')
ax.set_title('Total Moisture QT (kg/m^2)')
plt.colorbar(im, ax=ax)

# p_grid (最终降水，地理坐标)
ax = axes[1, 2]
im = ax.pcolormesh(X/1000, Y/1000, p_grid*1000*86400, 
                   shading='auto', cmap='Blues')
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('p_grid (mm/day) - Geographic')
plt.colorbar(im, ax=ax)
ax.annotate('', xy=(150, -80), xytext=(100, -80),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('debug_precipitation.png', dpi=150, bbox_inches='tight')
print("\nSaved: debug_precipitation.png")

# 沿s方向的剖面分析
center_t_idx = n_t_grid // 2
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# p_star 剖面
ax = axes[0, 0]
ax.plot(s/1000, p_star_pos_wind[:, center_t_idx]*1000*86400, 'b-', label='p_star')
ax.set_xlabel('s (km)')
ax.set_ylabel('Precipitation (mm/day)')
ax.set_title('p_star along s (center t)')
ax.axvline(0, color='k', linestyle='--', alpha=0.3)
ax.legend()

# f_v 剖面
ax = axes[0, 1]
ax.plot(s/1000, f_v_wind[:, center_t_idx], 'g-', label='f_v')
ax.set_xlabel('s (km)')
ax.set_ylabel('Vapor Ratio')
ax.set_title('Vapor ratio along s (center t)')
ax.axvline(0, color='k', linestyle='--', alpha=0.3)
ax.set_ylim(0, 1.1)
ax.legend()

# QT 剖面
ax = axes[1, 0]
ax.plot(s/1000, QT_star_pos_wind[:, center_t_idx], 'r-', label='QT')
ax.axhline(rho_s0*h_s, color='b', linestyle='--', label='rho_s0*h_s')
ax.set_xlabel('s (km)')
ax.set_ylabel('Moisture (kg/m^2)')
ax.set_title('Total moisture along s (center t)')
ax.axvline(0, color='k', linestyle='--', alpha=0.3)
ax.legend()

# 积分项剖面
ax = axes[1, 1]
ax.plot(s/1000, integral[:, center_t_idx], 'm-', label='integral')
ax.set_xlabel('s (km)')
ax.set_ylabel('Cumulative Integral')
ax.set_title('Cumulative integral along s')
ax.axvline(0, color='k', linestyle='--', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('debug_cross_section.png', dpi=150, bbox_inches='tight')
print("Saved: debug_cross_section.png")

plt.show()

print("\n" + "="*60)
print("Debug Complete")
print("="*60)
