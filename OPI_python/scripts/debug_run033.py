"""
调试 Run033 - 诊断双风场模拟问题
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from opi.get_input import tukey_window
from opi.coordinates import lonlat2xy
from opi.core.base_state import base_state
from opi.core.precipitation import precipitation_grid


def debug_precipitation():
    """调试降水计算"""
    print("="*60)
    print("调试降水计算")
    print("="*60)
    
    # 创建简单的测试地形
    x = np.linspace(-50000, 50000, 100)  # 100 km
    y = np.linspace(-50000, 50000, 100)
    X, Y = np.meshgrid(x, y)
    
    # 高斯山峰
    h_grid = 3000 * np.exp(-(X**2 + Y**2) / (2 * 15000**2))
    
    print(f"地形网格: {h_grid.shape}")
    print(f"高程范围: {h_grid.min():.0f} ~ {h_grid.max():.0f} m")
    
    # 参数（来自 run033）
    U = 6.6
    azimuth = 33.8
    T0 = 293.5
    M = 0.340
    kappa = 233368
    tau_c = 1296
    f_p0 = 1.0
    
    # 计算 buoyancy frequency
    h_max = np.max(h_grid)
    NM = M * U / h_max
    
    print(f"\n风场参数:")
    print(f"  U = {U:.1f} m/s")
    print(f"  Azimuth = {azimuth:.1f} deg")
    print(f"  T0 = {T0:.1f} K")
    print(f"  M = {M:.3f}")
    print(f"  NM = {NM*1000:.3f} mrad/s")
    print(f"  kappa = {kappa:.0f} m^2/s")
    print(f"  tau_c = {tau_c:.0f} s")
    
    # 计算大气基础状态
    base = base_state(NM, T0)
    
    print(f"\n基础状态:")
    print(f"  h_s = {base['h_s']:.0f} m")
    print(f"  h_rho = {base['h_rho']:.0f} m")
    print(f"  rho_s0 = {base['rho_s0']*1000:.2f} g/m^3")
    print(f"  gamma_ratio = {base['gamma_ratio']:.3f}")
    
    # 计算降水
    lat0 = 30.0
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    print(f"\nCoriolis: f_c = {f_c*1e4:.4f} x 10^-4 rad/s")
    
    result = precipitation_grid(
        x, y, h_grid, U, azimuth, NM, f_c,
        kappa, tau_c, base['h_rho'], base['z_bar'], base['T'],
        base['gamma_env'], base['gamma_sat'], base['gamma_ratio'],
        base['rho_s0'], base['h_s'], f_p0
    )
    
    p_grid = result['p_grid']
    
    print(f"\n降水结果:")
    print(f"  p_grid shape: {p_grid.shape}")
    print(f"  p_grid min: {p_grid.min():.6f}")
    print(f"  p_grid max: {p_grid.max():.6f}")
    print(f"  p_grid mean: {p_grid.mean():.6f}")
    print(f"  非零降水格点数: {np.sum(p_grid > 0)}")
    
    if np.max(p_grid) > 0:
        print(f"\n  [OK] 降水计算正常!")
    else:
        print(f"\n  [FAIL] 降水全为零 - 需要调试!")
    
    return result


def debug_two_winds():
    """调试双风场组合"""
    print("\n" + "="*60)
    print("调试双风场组合")
    print("="*60)
    
    # 简单地形
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 15000**2))
    
    # 风场 1
    U1, az1, T0_1, M1 = 6.6, 33.8, 293.5, 0.34
    kappa1, tau_c1, f_p0_1 = 233368, 1296, 1.0
    
    # 风场 2
    U2, az2, T0_2, M2 = 16.0, 125.7, 294.0, 0.69
    kappa2, tau_c2, f_p0_2 = 722120, 2450, 1.0
    
    fraction = 0.52
    
    lat0 = 30.0
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    # 风场 1
    h_max = np.max(h_grid)
    NM1 = M1 * U1 / h_max
    base1 = base_state(NM1, T0_1)
    
    result1 = precipitation_grid(
        x, y, h_grid, U1, az1, NM1, f_c,
        kappa1, tau_c1, base1['h_rho'], base1['z_bar'], base1['T'],
        base1['gamma_env'], base1['gamma_sat'], base1['gamma_ratio'],
        base1['rho_s0'], base1['h_s'], f_p0_1
    )
    
    p_grid_1 = result1['p_grid'] * fraction
    
    # 风场 2
    NM2 = M2 * U2 / h_max
    base2 = base_state(NM2, T0_2)
    
    result2 = precipitation_grid(
        x, y, h_grid, U2, az2, NM2, f_c,
        kappa2, tau_c2, base2['h_rho'], base2['z_bar'], base2['T'],
        base2['gamma_env'], base2['gamma_sat'], base2['gamma_ratio'],
        base2['rho_s0'], base2['h_s'], f_p0_2
    )
    
    p_grid_2 = result2['p_grid'] * (1 - fraction)
    
    print(f"风场 1 降水: min={p_grid_1.min():.6f}, max={p_grid_1.max():.6f}")
    print(f"风场 2 降水: min={p_grid_2.min():.6f}, max={p_grid_2.max():.6f}")
    
    p_grid = p_grid_1 + p_grid_2
    print(f"组合降水: min={p_grid.min():.6f}, max={p_grid.max():.6f}")
    
    # 检查网格形状
    print(f"\n网格形状:")
    print(f"  地形: {h_grid.shape}")
    print(f"  p_grid_1: {p_grid_1.shape}")
    print(f"  p_grid_2: {p_grid_2.shape}")
    print(f"  s_1: {result1['s'].shape}, t_1: {result1['t'].shape}")
    print(f"  s_2: {result2['s'].shape}, t_2: {result2['t'].shape}")
    
    return result1, result2


def main():
    print("Run033 调试脚本")
    print("="*60)
    
    # 测试降水计算
    result = debug_precipitation()
    
    # 测试双风场
    result1, result2 = debug_two_winds()
    
    print("\n" + "="*60)
    print("调试完成")
    print("="*60)


if __name__ == '__main__':
    main()
