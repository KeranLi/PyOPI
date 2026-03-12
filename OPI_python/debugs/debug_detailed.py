#!/usr/bin/env python
"""
详细调试降水计算，与MATLAB代码逐行对比
"""

import sys
sys.path.insert(0, 'OPI_python')

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid

# 加载MATLAB地形文件
topo_file = 'OPI Example Gaussian Mountain Range/data/EastDirectedGaussianTopography_3km height_lat45N.mat'

with h5py.File(topo_file, 'r') as f:
    h_grid = f['hGrid'][:].T
    lon = f['lon'][:].flatten()
    lat = f['lat'][:].flatten()

# 转换坐标
from opi.io import lonlat2xy
lon0 = (lon.min() + lon.max()) / 2
lat0 = (lat.min() + lat.max()) / 2
x, y = lonlat2xy(lon, lat, lon0, lat0)

# 参数（与MATLAB的run文件一致）
U = 10.0
azimuth = 90.0
T0 = 290.0
M = 0.25
NM = M * U / h_grid.max()
f_c = 1e-4
kappa = 0.0
tau_c = 1000.0
f_p0 = 1.0

print("="*70)
print("详细调试 - 与MATLAB代码逐行对比")
print("="*70)

print(f"\n=== 输入参数 ===")
print(f"U = {U}")
print(f"azimuth = {azimuth}")
print(f"T0 = {T0}")
print(f"M = {M}")
print(f"NM = {NM:.6e}")
print(f"f_c = {f_c}")
print(f"h_grid max = {h_grid.max():.2f} m")
print(f"h_grid shape = {h_grid.shape}")

# 基础状态
from opi.physics.thermodynamics import base_state
z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)

print(f"\n=== 基础状态输出 ===")
print(f"gamma_ratio = {gamma_ratio:.6f}")
print(f"rho_s0 = {rho_s0:.6e}")
print(f"h_s = {h_s:.2f}")
print(f"h_rho = {h_rho:.2f}")

# === fourier_solution ===
from opi.physics.fourier import wind_grid

Sxy, Txy, s, t, Xst, Yst = wind_grid(x, y, azimuth)

# 插值h_grid到风网格
from scipy.interpolate import RegularGridInterpolator
interp_func = RegularGridInterpolator(
    (y, x), 
    h_grid, 
    method='linear', 
    bounds_error=False, 
    fill_value=0
)
points = np.column_stack([Yst.ravel(), Xst.ravel()])
h_wind = interp_func(points).reshape(Xst.shape)

d_s = s[1] - s[0]
d_t = t[1] - t[0]
n_s, n_t = h_wind.shape

print(f"\n=== wind_grid输出 ===")
print(f"d_s = {d_s:.4f}")
print(f"d_t = {d_t:.4f}")
print(f"n_s = {n_s}, n_t = {n_t}")
print(f"h_wind max = {h_wind.max():.2f}")
print(f"h_wind sum = {h_wind.sum():.4e}")

# FFT参数
n_s_pad = 2 * n_s
n_t_pad = n_t

print(f"\n=== FFT参数 ===")
print(f"n_s_pad = {n_s_pad}")
print(f"n_t_pad = {n_t_pad}")

# 波数
i_k_s_most_neg = int(np.ceil(n_s_pad / 2)) + 1
k_s = np.arange(n_s_pad) / n_s_pad
k_s[i_k_s_most_neg:n_s_pad] = k_s[i_k_s_most_neg:n_s_pad] - 1
k_s = 2 * np.pi * k_s / d_s
d_k_s = k_s[1] - k_s[0]

i_k_t_most_neg = int(np.ceil(n_t_pad / 2)) + 1
k_t = np.arange(n_t_pad) / n_t_pad
k_t[i_k_t_most_neg:n_t_pad] = k_t[i_k_t_most_neg:n_t_pad] - 1
k_t = 2 * np.pi * k_t / d_t

print(f"\n=== 波数 ===")
print(f"d_k_s = {d_k_s:.6e}")
print(f"k_s[0:5] = {k_s[:5]}")
print(f"k_s[-5:] = {k_s[-5:]}")

# FFT
h_hat = np.fft.fft2(h_wind, s=(n_s_pad, n_t_pad))

print(f"\n=== h_hat (FFT) ===")
print(f"h_hat[0,0] = {h_hat[0,0]:.4e}")
print(f"max |h_hat| = {np.abs(h_hat).max():.4e}")

# k_z计算
denominator = (U * k_s.reshape(-1, 1))**2 - f_c**2
i_zero = np.abs(denominator) < (U * d_k_s / 2)**2
# 广播i_zero到h_hat的形状
i_zero_broadcast = np.broadcast_to(i_zero, h_hat.shape)
h_hat[i_zero_broadcast] = 0
denominator[denominator == 0] = np.finfo(float).eps

k_z_sq = (k_s.reshape(-1, 1)**2 + k_t**2) * ((NM**2 - (U * k_s.reshape(-1, 1))**2) / denominator) - \
         1.0 / (4 * h_rho**2)
k_z_sq = k_z_sq.astype(np.complex128)
k_z = np.sqrt(k_z_sq)

i_neg = (np.real(k_z_sq) > 0) & (k_s.reshape(-1, 1) < 0)
k_z[i_neg] = -k_z[i_neg]

print(f"\n=== k_z ===")
print(f"k_z[0,0] = {k_z[0,0]:.4e}")
print(f"min Re(k_z) = {np.real(k_z).min():.4e}")
print(f"max Re(k_z) = {np.real(k_z).max():.4e}")

# === precipitation_grid ===

# isotherm 258K
from opi.physics.precipitation import isotherm
z258_wind = isotherm(258.0, z_bar, T, gamma_env, gamma_sat, h_rho, 
                     n_s, n_t, h_hat, k_z)

print(f"\n=== z258_wind ===")
print(f"min = {z258_wind.min():.2f}")
print(f"max = {z258_wind.max():.2f}")

# tau_f
w_f_snow = -1.0
w_f_rain = -6.0
tau_f_grid = np.where(
    z258_wind <= h_wind,
    -h_s / w_f_snow,
    -(z258_wind - h_wind) / w_f_rain - h_s * np.exp(-(z258_wind - h_wind) / h_s) / w_f_snow
)
tau_f = np.mean(tau_f_grid)

print(f"\n=== tau_f ===")
print(f"tau_f = {tau_f:.2f}")

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
print(f"max |GS_hat| = {np.abs(GS_hat).max():.6e}")
print(f"max |GC_hat| = {np.abs(GC_hat).max():.6f}")
print(f"max |GF_hat| = {np.abs(GF_hat).max():.6f}")

# p_star_hat
p_star_hat = GS_hat * GC_hat * GF_hat * h_hat

print(f"\n=== p_star_hat ===")
print(f"max |p_star_hat| = {np.abs(p_star_hat).max():.6e}")

# IFFT
p_star_pos_wind = np.fft.ifft2(p_star_hat)
p_star_pos_wind = np.real(p_star_pos_wind)
p_star_pos_wind = p_star_pos_wind[:n_s, :n_t]
p_star_pos_wind[p_star_pos_wind < 0] = 0

print(f"\n=== p_star_pos_wind (IFFT后) ===")
print(f"max = {p_star_pos_wind.max():.6e} kg/m^2/s")
print(f"max (mm/day) = {p_star_pos_wind.max()*1000*86400:.2f}")
print(f"mean = {p_star_pos_wind.mean():.6e} kg/m^2/s")

# 柱密度
QC_star_pos_wind = np.fft.ifft2(tau_c * GS_hat * GC_hat * h_hat)
QC_star_pos_wind = np.real(QC_star_pos_wind)
QC_star_pos_wind = QC_star_pos_wind[:n_s, :n_t]
QC_star_pos_wind[QC_star_pos_wind < 0] = 0

QF_star_pos_wind = np.fft.ifft2(tau_f * GS_hat * GC_hat * GF_hat * h_hat)
QF_star_pos_wind = np.real(QF_star_pos_wind)
QF_star_pos_wind = QF_star_pos_wind[:n_s, :n_t]
QF_star_pos_wind[QF_star_pos_wind < 0] = 0

QT_star_pos_wind = rho_s0 * h_s + QC_star_pos_wind + QF_star_pos_wind

print(f"\n=== 柱密度 ===")
print(f"QC_star max = {QC_star_pos_wind.max():.4f}")
print(f"QF_star max = {QF_star_pos_wind.max():.4f}")
print(f"QT_star max = {QT_star_pos_wind.max():.4f}")
print(f"rho_s0 * h_s = {rho_s0 * h_s:.4f}")

# 水汽比例
integrand = p_star_pos_wind / QT_star_pos_wind
integral = cumulative_trapezoid(integrand, dx=d_s, axis=0, initial=0)
f_v_wind = (rho_s0 * h_s / QT_star_pos_wind) * np.exp(-(1.0/U) * integral)

print(f"\n=== f_v_wind ===")
print(f"min = {f_v_wind.min():.6f}")
print(f"max = {f_v_wind.max():.6f}")
print(f"mean = {f_v_wind.mean():.6f}")

# 最终降水
p_wind = f_v_wind * p_star_pos_wind

print(f"\n=== p_wind (最终降水) ===")
print(f"max = {p_wind.max():.6e} kg/m^2/s")
print(f"max (mm/day) = {p_wind.max()*1000*86400:.2f}")

# 插值回地理网格
interp_func = RegularGridInterpolator((s, t), p_wind, method='linear', bounds_error=False, fill_value=0)
p_grid = interp_func(np.column_stack([Sxy.ravel(), Txy.ravel()])).reshape(Sxy.shape)

print(f"\n=== p_grid (地理坐标) ===")
print(f"shape = {p_grid.shape}")
print(f"max = {p_grid.max():.6e} kg/m^2/s")
print(f"max (mm/day) = {p_grid.max()*1000*86400:.2f}")

print("\n" + "="*70)
print("关键问题：降水率过高！")
print(f"当前值: {p_grid.max()*1000*86400:.2f} mm/day")
print("预期值: ~10-50 mm/day (Smith & Barstad 2004)")
print("="*70)
