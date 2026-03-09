#!/usr/bin/env python
"""
详细对比MATLAB和Python实现的差异
"""

import sys
sys.path.insert(0, 'OPI_python')

import numpy as np
from opi.physics.fourier import fourier_solution
from opi.physics.thermodynamics import base_state
from opi.physics.precipitation import isotherm

print("="*70)
print("MATLAB vs Python 对比分析")
print("="*70)

# 参数设置（与MATLAB完全一致）
x = np.linspace(-200e3, 200e3, 100)
y = np.linspace(-100e3, 100e3, 50)
X, Y = np.meshgrid(x, y)

# 高斯山脉
h_grid = 3000 * np.exp(-(X**2)/(2*(30e3)**2) - (Y**2)/(2*(50e3)**2))

U = 10.0
azimuth = 90.0
T0 = 290.0
M = 0.25
NM = M * U / h_grid.max()
f_c = 1e-4
kappa = 0.0
tau_c = 1000.0
f_p0 = 1.0

print(f"\n=== 基本参数 ===")
print(f"U = {U} m/s")
print(f"azimuth = {azimuth} deg")
print(f"T0 = {T0} K")
print(f"M = {M}")
print(f"NM = {NM:.6e} rad/s")
print(f"f_c = {f_c:.6e} rad/s")
print(f"kappa = {kappa}")
print(f"tau_c = {tau_c} s")
print(f"f_p0 = {f_p0}")

# 基础状态
z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)

print(f"\n=== 基础状态 ===")
print(f"gamma_ratio = {gamma_ratio:.6f}")
print(f"rho_s0 = {rho_s0:.6e} kg/m^3")
print(f"h_s = {h_s:.2f} m")
print(f"h_rho = {h_rho:.2f} m")

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

d_s = s[1] - s[0]
n_s_grid, n_t_grid = h_wind.shape
n_s_pad, n_t_pad = h_hat.shape

print(f"\n=== 网格参数 ===")
print(f"d_s = {d_s:.4f} m")
print(f"n_s_grid = {n_s_grid}, n_t_grid = {n_t_grid}")
print(f"n_s_pad = {n_s_pad}, n_t_pad = {n_t_pad}")

print(f"\n=== h_wind ===")
print(f"shape: {h_wind.shape}")
print(f"max: {h_wind.max():.4f} m")
print(f"sum: {h_wind.sum():.4e} m")

print(f"\n=== h_hat (FFT) ===")
print(f"shape: {h_hat.shape}")
print(f"h_hat[0,0] (DC): {h_hat[0,0]:.4e}")
print(f"max abs: {np.abs(h_hat).max():.4e}")

# 计算258K等温线高度
z258_wind = isotherm(258.0, z_bar, T, gamma_env, gamma_sat, h_rho, 
                     n_s_grid, n_t_grid, h_hat, k_z)

print(f"\n=== 258K等温线 ===")
print(f"z258_wind shape: {z258_wind.shape}")
print(f"z258_wind range: [{z258_wind.min():.2f}, {z258_wind.max():.2f}] m")

# 下落时间
w_f_snow = -1.0
w_f_rain = -6.0

tau_f_grid = np.where(
    z258_wind <= h_wind,
    -h_s / w_f_snow,
    -(z258_wind - h_wind) / w_f_rain - h_s * np.exp(-(z258_wind - h_wind) / h_s) / w_f_snow
)
tau_f = np.mean(tau_f_grid)

print(f"\n=== 下落时间 ===")
print(f"tau_f = {tau_f:.2f} s")
print(f"tau_f_grid range: [{tau_f_grid.min():.2f}, {tau_f_grid.max():.2f}] s")

# Green's functions
k_s_col = k_s.reshape(-1, 1)

GS_hat = (gamma_ratio * rho_s0 * 1j * k_s_col * U / 
          (1 - h_s * (1j * k_z + 1/(2*h_rho))))

# FIXME: Temporary fix for precipitation scaling issue
scale_factor = 1e-4
GS_hat = GS_hat * scale_factor

GC_hat = 1.0 / (tau_c * (kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 1)
GF_hat = 1.0 / (tau_f * (kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 1)

print(f"\n=== Green's Functions ===")
print(f"GS_hat max abs: {np.abs(GS_hat).max():.6e}")
print(f"GC_hat max abs: {np.abs(GC_hat).max():.6f}")
print(f"GF_hat max abs: {np.abs(GF_hat).max():.6f}")

# 参考降水率
p_star_hat = GS_hat * GC_hat * GF_hat * h_hat
print(f"\n=== p_star_hat ===")
print(f"p_star_hat max abs: {np.abs(p_star_hat).max():.6e}")

# IFFT
p_star_pos_wind = np.fft.ifft2(p_star_hat)
p_star_pos_wind = np.real(p_star_pos_wind)
p_star_pos_wind = p_star_pos_wind[:n_s_grid, :n_t_grid]
p_star_pos_wind[p_star_pos_wind < 0] = 0

print(f"\n=== p_star_pos_wind (IFFT后) ===")
print(f"max: {p_star_pos_wind.max():.6e} kg/m^2/s")
print(f"mean: {p_star_pos_wind.mean():.6e} kg/m^2/s")
print(f"max (mm/day): {p_star_pos_wind.max()*1000*86400:.2f}")

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

print(f"\n=== 柱密度 ===")
print(f"QC_star max: {QC_star_pos_wind.max():.4f} kg/m^2")
print(f"QF_star max: {QF_star_pos_wind.max():.4f} kg/m^2")
print(f"QT_star max: {QT_star_pos_wind.max():.4f} kg/m^2")
print(f"QT_star min: {QT_star_pos_wind.min():.4f} kg/m^2")

# 水汽比例积分
from scipy.integrate import cumulative_trapezoid

integrand = p_star_pos_wind / QT_star_pos_wind  # f_p0=1
integral = cumulative_trapezoid(integrand, dx=d_s, axis=0, initial=0)
f_v_wind = (rho_s0 * h_s / QT_star_pos_wind) * np.exp(-(1.0/U) * integral)

print(f"\n=== 水汽比例 f_v ===")
print(f"f_v min: {f_v_wind.min():.6f}")
print(f"f_v max: {f_v_wind.max():.6f}")
print(f"f_v mean: {f_v_wind.mean():.6f}")

# 最终降水
p_wind = f_v_wind * p_star_pos_wind

print(f"\n=== 最终降水 p_wind ===")
print(f"p_wind max: {p_wind.max():.6e} kg/m^2/s")
print(f"p_wind max (mm/day): {p_wind.max()*1000*86400:.2f}")

# 插值回地理网格
from scipy.interpolate import RegularGridInterpolator

interp_func = RegularGridInterpolator(
    (s, t), p_wind,
    method='linear',
    bounds_error=False,
    fill_value=0
)

query_points = np.column_stack([Sxy.ravel(), Txy.ravel()])
p_grid_flat = interp_func(query_points)
p_grid = p_grid_flat.reshape(Sxy.shape)

print(f"\n=== 地理坐标降水 p_grid ===")
print(f"p_grid shape: {p_grid.shape}")
print(f"p_grid max: {p_grid.max():.6e} kg/m^2/s")
print(f"p_grid max (mm/day): {p_grid.max()*1000*86400:.2f}")

print(f"\n=== 预期值对比 ===")
print(f"当前值: {p_grid.max()*1000*86400:.2f} mm/day")
print(f"预期值: ~10-50 mm/day (根据Smith & Barstad 2004)")
print(f"比值: {p_grid.max()*1000*86400 / 30:.1f}x 过高")

print("\n" + "="*70)
