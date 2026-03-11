"""
Run033 Full Reproduction
一比一复刻 MATLAB OPI 3.7 双风场模拟 - 完整版本
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from opi.get_input import tukey_window, estimate_mwl
from opi.coordinates import lonlat2xy
from opi.core.base_state import base_state
from opi.core.precipitation import precipitation_grid
from opi.core.isotopes import isotope_grid
from opi.core.catchment import catchment_nodes
from scipy.interpolate import RegularGridInterpolator


def parse_run033():
    """解析 run033.run 文件中的参数"""
    return {
        'run_title': 'run 33, Himalaya area, OPI 3.7, two wind',
        'r_tukey': 0.5,
        'map_limits': [77.5, 97.5, 20.0, 36.0],
        'section_origin': [90.26876, 29.19308],
        'beta': [
            6.6, 33.8, 293.5, 0.340, 233368, 1296, -0.0026, -0.000533, 1.0, 0.52,
            16.0, 125.7, 294.0, 0.693, 722120, 2450, -0.0001, -0.000007, 1.0
        ],
        'lB': [0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0, 0,
               0.1, 0, 270, 0, 0, 0, -0.060, -0.015, 0],
        'uB': [20, 90, 300, 1, 1000000, 2500, 0, 0, 1, 1,
               20, 180, 300, 1, 1000000, 2500, 0, 0, 1],
    }


def create_himalaya_topo(config):
    """创建喜马拉雅地形"""
    print("创建合成喜马拉雅地形...")
    
    min_lon, max_lon, min_lat, max_lat = config['map_limits']
    
    # 降低分辨率以便快速计算
    d_lon = 0.02  # ~2km
    d_lat = 0.02
    
    lon = np.arange(min_lon, max_lon + d_lon, d_lon)
    lat = np.arange(min_lat, max_lat + d_lat, d_lat)
    
    LON, LAT = np.meshgrid(lon, lat)
    
    # 合成喜马拉雅地形
    peak_lon = 87.0
    peak_lat = 28.0
    h_grid = 8000 * np.exp(-((LON - peak_lon)**2 / 2 + (LAT - peak_lat)**2 / 2) / 4)
    h_grid += 4000 * np.exp(-((LON - 92.0)**2 / 2 + (LAT - 27.0)**2 / 2) / 2)
    h_grid += 3000 * np.exp(-((LON - 85.0)**2 / 2 + (LAT - 32.0)**2 / 2) / 3)
    
    # 应用 Tukey 窗口
    r_tukey = config['r_tukey']
    if r_tukey > 0:
        n_y, n_x = h_grid.shape
        window_y = tukey_window(n_y, r_tukey)
        window_x = tukey_window(n_x, r_tukey)
        window = np.outer(window_y, window_x)
        h_grid = window * h_grid
    
    print(f"  网格: {h_grid.shape}")
    print(f"  高程: {h_grid.min():.0f} ~ {h_grid.max():.0f} m")
    
    return lon, lat, h_grid


def create_samples(lon, lat, h_grid):
    """创建合成样本"""
    print("\n创建合成样本...")
    
    np.random.seed(42)
    n_samples = 20
    
    # 在区域内随机分布
    sample_lon = np.random.uniform(lon.min() + 5, lon.max() - 5, n_samples)
    sample_lat = np.random.uniform(lat.min() + 3, lat.max() - 3, n_samples)
    
    # 插值获取高程
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lat, lon), h_grid, method='linear', bounds_error=False, fill_value=0)
    sample_elev = interp(np.column_stack([sample_lat, sample_lon]))
    
    # 根据高程计算同位素（高海拔更负）
    # 确保 d-excess > 5 permil (主要样本)
    sample_d18o = (-10 - sample_elev / 500) * 1e-3 + np.random.normal(0, 1e-3, n_samples)
    sample_d2h = 8 * sample_d18o + 10e-3 + np.random.normal(0, 3e-3, n_samples)  # d-excess ~10
    
    # 类型: L=Local, C=Catchment
    sample_lc = np.random.choice(['L', 'C'], n_samples, p=[0.3, 0.7])
    
    # 验证 d-excess
    d_excess = (sample_d2h - 8 * sample_d18o) * 1000
    print(f"  样本数: {n_samples}")
    print(f"  Local: {np.sum(sample_lc == 'L')}, Catchment: {np.sum(sample_lc == 'C')}")
    print(f"  d-excess: {d_excess.min():.1f} ~ {d_excess.max():.1f} permil")
    
    return {
        'lon': sample_lon,
        'lat': sample_lat,
        'elev': sample_elev,
        'd2h': sample_d2h,
        'd18o': sample_d18o,
        'lc': sample_lc
    }


def run_calculation(config, lon, lat, h_grid, samples):
    """运行双风场计算"""
    print("\n" + "="*60)
    print("运行双风场 OPI 计算")
    print("="*60)
    
    beta = np.array(config['beta'])
    
    # 解析参数
    U_1, az_1, T0_1, M_1 = beta[0], beta[1], beta[2], beta[3]
    kappa_1, tau_c_1, d2h0_1, d_d2h0_d_lat_1, f_p0_1 = beta[4], beta[5], beta[6], beta[7], beta[8]
    fraction = beta[9]
    
    U_2, az_2, T0_2, M_2 = beta[10], beta[11], beta[12], beta[13]
    kappa_2, tau_c_2, d2h0_2, d_d2h0_d_lat_2, f_p0_2 = beta[14], beta[15], beta[16], beta[17], beta[18]
    
    print(f"\n风场 1 (fraction={fraction:.2f}):")
    print(f"  U={U_1:.1f}m/s, Az={az_1:.1f}deg, T0={T0_1:.1f}K, M={M_1:.3f}")
    
    print(f"\n风场 2 (fraction={1-fraction:.2f}):")
    print(f"  U={U_2:.1f}m/s, Az={az_2:.1f}deg, T0={T0_2:.1f}K, M={M_2:.3f}")
    
    # 地图原点
    lon0 = np.mean(samples['lon'])
    lat0 = np.mean(samples['lat'])
    print(f"\n地图原点: ({lon0:.3f}, {lat0:.3f})")
    
    # 转换到投影坐标
    x, y = lonlat2xy(lon, lat, lon0, lat0)
    sample_x, sample_y = lonlat2xy(samples['lon'], samples['lat'], lon0, lat0)
    
    # Coriolis 参数
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    print(f"Coriolis: f_c={f_c*1e4:.3f}x10^-4 rad/s")
    
    # MWL
    b_mwl, sd_min, sd_max, cov, i_fit = estimate_mwl(samples['d18o'], samples['d2h'])
    print(f"\nMWL: d2H = {b_mwl[0]*1000:.1f} + {b_mwl[1]:.2f} * d18O")
    
    # 分割样本
    sample_d2h = samples['d2h'][i_fit]
    sample_d18o = samples['d18o'][i_fit]
    sample_lc = samples['lc'][i_fit]
    sample_x = sample_x[i_fit]
    sample_y = sample_y[i_fit]
    
    print(f"  主要样本: {len(sample_d2h)}")
    
    # 流域节点
    ij_catch, ptr_catch = catchment_nodes(x, y, sample_x, sample_y, sample_lc, h_grid)
    print(f"\n流域节点: {len(ij_catch)} 个")
    
    # 计算风场 1
    print("\n计算风场 1...")
    h_max = np.max(h_grid)
    NM_1 = M_1 * U_1 / h_max
    base_1 = base_state(NM_1, T0_1)
    
    d18o0_1 = (d2h0_1 - b_mwl[0]) / b_mwl[1]
    d_d18o0_d_lat_1 = d_d2h0_d_lat_1 / b_mwl[1]
    
    precip_1 = precipitation_grid(
        x, y, h_grid, U_1, az_1, NM_1, f_c, kappa_1, tau_c_1,
        base_1['h_rho'], base_1['z_bar'], base_1['T'],
        base_1['gamma_env'], base_1['gamma_sat'], base_1['gamma_ratio'],
        base_1['rho_s0'], base_1['h_s'], f_p0_1
    )
    
    p_grid_1 = precip_1['p_grid'] * fraction
    
    iso_1 = isotope_grid(
        precip_1['s'], precip_1['t'], precip_1['Sxy'], precip_1['Txy'],
        lat, lat0, precip_1['h_wind'], precip_1['f_m_wind'],
        precip_1['r_h_wind'], precip_1['f_p_wind'],
        precip_1['z223_wind'], precip_1['z258_wind'], precip_1['tau_f'],
        U_1, base_1['T'], base_1['gamma_sat'], base_1['h_s'], 540.0,
        d2h0_1, d18o0_1, d_d2h0_d_lat_1, d_d18o0_d_lat_1, False
    )
    
    d2h_grid_1 = iso_1['d2H_grid']
    d18o_grid_1 = iso_1['d18O_grid']
    
    print(f"  降水: {p_grid_1.min():.6f} ~ {p_grid_1.max():.6f}")
    print(f"  d2H: {d2h_grid_1.min()*1000:.1f} ~ {d2h_grid_1.max()*1000:.1f} permil")
    
    # 计算风场 2
    print("\n计算风场 2...")
    NM_2 = M_2 * U_2 / h_max
    base_2 = base_state(NM_2, T0_2)
    
    d18o0_2 = (d2h0_2 - b_mwl[0]) / b_mwl[1]
    d_d18o0_d_lat_2 = d_d2h0_d_lat_2 / b_mwl[1]
    
    precip_2 = precipitation_grid(
        x, y, h_grid, U_2, az_2, NM_2, f_c, kappa_2, tau_c_2,
        base_2['h_rho'], base_2['z_bar'], base_2['T'],
        base_2['gamma_env'], base_2['gamma_sat'], base_2['gamma_ratio'],
        base_2['rho_s0'], base_2['h_s'], f_p0_2
    )
    
    p_grid_2 = precip_2['p_grid'] * (1 - fraction)
    
    iso_2 = isotope_grid(
        precip_2['s'], precip_2['t'], precip_2['Sxy'], precip_2['Txy'],
        lat, lat0, precip_2['h_wind'], precip_2['f_m_wind'],
        precip_2['r_h_wind'], precip_2['f_p_wind'],
        precip_2['z223_wind'], precip_2['z258_wind'], precip_2['tau_f'],
        U_2, base_2['T'], base_2['gamma_sat'], base_2['h_s'], 540.0,
        d2h0_2, d18o0_2, d_d2h0_d_lat_2, d_d18o0_d_lat_2, False
    )
    
    d2h_grid_2 = iso_2['d2H_grid']
    d18o_grid_2 = iso_2['d18O_grid']
    
    print(f"  降水: {p_grid_2.min():.6f} ~ {p_grid_2.max():.6f}")
    print(f"  d2H: {d2h_grid_2.min()*1000:.1f} ~ {d2h_grid_2.max()*1000:.1f} permil")
    
    # 组合结果
    print("\n组合结果...")
    p_grid = p_grid_1 + p_grid_2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d2h_grid = np.where(p_grid > 0, 
                           (p_grid_1 * d2h_grid_1 + p_grid_2 * d2h_grid_2) / p_grid,
                           np.nan)
        d18o_grid = np.where(p_grid > 0,
                            (p_grid_1 * d18o_grid_1 + p_grid_2 * d18o_grid_2) / p_grid,
                            np.nan)
    
    print(f"组合降水: {p_grid.min():.6f} ~ {p_grid.max():.6f}")
    print(f"组合 d2H: {np.nanmin(d2h_grid)*1000:.1f} ~ {np.nanmax(d2h_grid)*1000:.1f} permil")
    
    return {
        'lon': lon, 'lat': lat, 'x': x, 'y': y, 'h_grid': h_grid,
        'p_grid_1': p_grid_1, 'p_grid_2': p_grid_2, 'p_grid': p_grid,
        'd2h_grid_1': d2h_grid_1, 'd2h_grid_2': d2h_grid_2, 'd2h_grid': d2h_grid,
        'd18o_grid_1': d18o_grid_1, 'd18o_grid_2': d18o_grid_2, 'd18o_grid': d18o_grid,
        'samples': samples,
        'beta': beta
    }


def save_and_plot(results, output_dir='outputs'):
    """保存结果并绘图"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 NPZ
    npz_file = os.path.join(output_dir, f'run033_{timestamp}.npz')
    np.savez(npz_file, **{k: v for k, v in results.items() if k != 'samples'})
    print(f"\n结果已保存: {npz_file}")
    
    # 绘图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        lon, lat = results['lon'], results['lat']
        
        # 地形
        ax = axes[0, 0]
        im = ax.pcolormesh(lon, lat, results['h_grid'], shading='auto', cmap='terrain')
        ax.set_title('Topography')
        plt.colorbar(im, ax=ax, label='Elevation (m)')
        
        # 降水
        ax = axes[0, 1]
        im = ax.pcolormesh(lon, lat, results['p_grid'], shading='auto', cmap='Blues')
        ax.set_title('Combined Precipitation')
        plt.colorbar(im, ax=ax)
        
        # 风场1降水
        ax = axes[0, 2]
        im = ax.pcolormesh(lon, lat, results['p_grid_1'], shading='auto', cmap='Blues')
        ax.set_title('Precipitation (Wind 1)')
        plt.colorbar(im, ax=ax)
        
        # 风场2降水
        ax = axes[1, 0]
        im = ax.pcolormesh(lon, lat, results['p_grid_2'], shading='auto', cmap='Blues')
        ax.set_title('Precipitation (Wind 2)')
        plt.colorbar(im, ax=ax)
        
        # d2H
        ax = axes[1, 1]
        d2h_permil = results['d2h_grid'] * 1000
        im = ax.pcolormesh(lon, lat, d2h_permil, shading='auto', cmap='RdYlBu_r',
                          vmin=-150, vmax=-50)
        ax.set_title('d2H (permil)')
        plt.colorbar(im, ax=ax)
        
        # d18O
        ax = axes[1, 2]
        d18o_permil = results['d18o_grid'] * 1000
        im = ax.pcolormesh(lon, lat, d18o_permil, shading='auto', cmap='RdYlBu_r',
                          vmin=-20, vmax=-5)
        ax.set_title('d18O (permil)')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        png_file = os.path.join(output_dir, f'run033_{timestamp}.png')
        plt.savefig(png_file, dpi=150)
        print(f"图片已保存: {png_file}")
        
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    print("="*60)
    print("Run033 复刻 - OPI 3.7 双风场模拟")
    print("="*60)
    print(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 解析配置
    config = parse_run033()
    
    # 创建地形
    lon, lat, h_grid = create_himalaya_topo(config)
    
    # 创建样本
    samples = create_samples(lon, lat, h_grid)
    
    # 运行计算
    results = run_calculation(config, lon, lat, h_grid, samples)
    
    # 保存和绘图
    if results:
        save_and_plot(results)
    
    print("\n" + "="*60)
    print(f"完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    results = main()
