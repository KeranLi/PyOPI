"""
Run033 Replica - 一比一复刻 MATLAB opiCalc_TwoWinds

严格仿照 MATLAB OPI 3.7 的 run033 模拟过程
使用真实数据：Himalaya 地形和同位素样本
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
from scipy.io import loadmat
import h5py

# 添加项目路径
script_dir = Path(__file__).parent.resolve()
opi_root = script_dir.parent
sys.path.insert(0, str(opi_root))

# Import OPI 模块
from opi import (
    base_state,
    precipitation_grid,
    isotope_grid,
    lonlat2xy,
    catchment_nodes,
    grid_read,
    get_input,
)


def find_data_file(filename, search_paths):
    """在多个路径中查找数据文件"""
    for path in search_paths:
        full_path = Path(path) / filename
        if full_path.exists():
            return str(full_path)
    return None


def parse_run_file(run_file_path):
    """解析 run 文件 - 仿照 MATLAB getRunFile
    
    Run033 参数（来自 MATLAB run033.run）：
    - Wind1: U=6.6, Az=33.8°, T0=293.5K, M=0.34, kappa=233368, tau_c=1296
    - Wind2: U=16.0, Az=125.7°, T0=294K, M=0.693, kappa=722120, tau_c=2450
    - Fraction: 0.52
    """
    config = {
        'run_title': 'run 33, Himalaya area, OPI 3.7, two wind',
        'topo_file': 'Himalaya topography_April 2023_Gebco1kmMakima.mat',
        'r_tukey': 0.5,
        'sample_file': 'Himalaya Water Isotopes 8 March 2023.xlsx',
        'beta': np.array([
            6.6, 33.8, 293.5, 0.340, 233368, 1296, -0.0026, -0.000533, 1.0, 0.52,
            16.0, 125.7, 294.0, 0.693, 722120, 2450, -0.0001, -0.000007, 1.0
        ]),
        'lB': np.array([0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0, 0,
                       0.1, 0, 270, 0, 0, 0, -0.060, -0.015, 0]),
        'uB': np.array([20, 90, 300, 1, 1000000, 2500, 0, 0, 1, 1,
                       20, 180, 300, 1, 1000000, 2500, 0, 0, 1])
    }
    return config


def estimate_mwl(d18o, d2h, sd_res_ratio=28.3):
    """估算气象水线 (Meteoric Water Line)"""
    d18o = np.asarray(d18o).flatten()
    d2h = np.asarray(d2h).flatten()
    
    # 简单的线性回归
    n = len(d18o)
    x_mean = np.mean(d18o)
    y_mean = np.mean(d2h)
    
    ss_xy = np.sum((d18o - x_mean) * (d2h - y_mean))
    ss_xx = np.sum((d18o - x_mean) ** 2)
    
    slope = ss_xy / ss_xx if ss_xx > 0 else 8.0
    intercept = y_mean - slope * x_mean
    
    # 计算残差标准差
    residuals = d2h - (slope * d18o + intercept)
    sd_residuals = np.std(residuals)
    
    # 构建协方差矩阵
    cov = np.array([[sd_residuals**2, 0], [0, sd_residuals**2]])
    
    return np.array([intercept, slope]), sd_residuals, sd_residuals, cov, np.arange(n)


def calc_two_winds(beta, f_c, h_r, x, y, lat, lat0, h_grid, b_mwl_sample,
                   ij_catch, ptr_catch, sample_d2h, sample_d18o, cov,
                   n_parameters_free, is_fit=False):
    """双风场计算 - 仿照 MATLAB calc_TwoWinds"""
    
    # 提取参数
    U_1, azimuth_1, T0_1, M_1 = beta[0], beta[1], beta[2], beta[3]
    kappa_1, tau_c_1, d2h0_1, d_d2h0_d_lat_1, f_p0_1 = beta[4], beta[5], beta[6], beta[7], beta[8]
    fraction = beta[9]
    U_2, azimuth_2, T0_2, M_2 = beta[10], beta[11], beta[12], beta[13]
    kappa_2, tau_c_2, d2h0_2, d_d2h0_d_lat_2, f_p0_2 = beta[14], beta[15], beta[16], beta[17], beta[18]
    
    # 计算无量纲参数
    h_max = np.max(h_grid)
    NM_1 = M_1 * U_1 / h_max
    NM_2 = M_2 * U_2 / h_max
    
    n_samples = len(ptr_catch) - 1 if len(ptr_catch) > 0 else 0
    
    print(f"\nWind 1: U={U_1:.1f} m/s, Az={azimuth_1:.1f}°, M={M_1:.3f}, NM={NM_1:.6f}")
    print(f"Wind 2: U={U_2:.1f} m/s, Az={azimuth_2:.1f}°, M={M_2:.3f}, NM={NM_2:.6f}")
    print(f"Fraction: {fraction:.3f}")
    
    # ============ 风场 1 ============
    z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1, rho_s0_1, h_s_1, rho0_1, h_rho_1 = \
        base_state(NM_1, T0_1)
    
    d18o0_1 = (d2h0_1 - b_mwl_sample[0]) / b_mwl_sample[1]
    d_d18o0_d_lat_1 = d_d2h0_d_lat_1 / b_mwl_sample[1]
    
    print("  Computing precipitation grid for wind 1...")
    precip_1 = precipitation_grid(
        x, y, h_grid, U_1, azimuth_1, NM_1, f_c, kappa_1, tau_c_1, h_rho_1,
        z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1, rho_s0_1, h_s_1, f_p0_1
    )
    
    p_grid_1 = precip_1['p_grid'] * fraction
    
    print("  Computing isotope grid for wind 1...")
    iso_1 = isotope_grid(
        precip_1['s'], precip_1['t'], precip_1['Sxy'], precip_1['Txy'],
        lat, lat0, precip_1['h_wind'], precip_1['f_m_wind'],
        precip_1['r_h_wind'], precip_1['f_p_wind'],
        precip_1['z223_wind'], precip_1['z258_wind'], precip_1['tau_f'],
        U_1, T_1, gamma_sat_1, h_s_1, h_r, d2h0_1, d18o0_1,
        d_d2h0_d_lat_1, d_d18o0_d_lat_1, is_fit
    )
    
    d2h_grid_1 = iso_1['d2H_grid']
    d18o_grid_1 = iso_1['d18O_grid']
    tau_f_1 = precip_1['tau_f']
    
    # 样本预测 1
    d2h_pred_1 = np.zeros(n_samples)
    d18o_pred_1 = np.zeros(n_samples)
    p_sum_pred_1 = np.zeros(n_samples)
    
    for k in range(n_samples):
        ij = ij_catch[ptr_catch[k]:ptr_catch[k+1]] if k < len(ptr_catch) - 1 else ij_catch[ptr_catch[k]:]
        p_sum_pred_1[k] = sum(p_grid_1[i, j] for i, j in ij)
        if p_sum_pred_1[k] > 0:
            wt = np.array([p_grid_1[i, j] / p_sum_pred_1[k] for i, j in ij])
            d2h_pred_1[k] = sum(wt[m] * d2h_grid_1[i, j] for m, (i, j) in enumerate(ij))
            d18o_pred_1[k] = sum(wt[m] * d18o_grid_1[i, j] for m, (i, j) in enumerate(ij))
    
    # ============ 风场 2 ============
    z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2, rho_s0_2, h_s_2, rho0_2, h_rho_2 = \
        base_state(NM_2, T0_2)
    
    d18o0_2 = (d2h0_2 - b_mwl_sample[0]) / b_mwl_sample[1]
    
    print("  Computing precipitation grid for wind 2...")
    precip_2 = precipitation_grid(
        x, y, h_grid, U_2, azimuth_2, NM_2, f_c, kappa_2, tau_c_2, h_rho_2,
        z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2, rho_s0_2, h_s_2, f_p0_2
    )
    
    p_grid_2 = precip_2['p_grid'] * (1 - fraction)
    
    print("  Computing isotope grid for wind 2...")
    iso_2 = isotope_grid(
        precip_2['s'], precip_2['t'], precip_2['Sxy'], precip_2['Txy'],
        lat, lat0, precip_2['h_wind'], precip_2['f_m_wind'],
        precip_2['r_h_wind'], precip_2['f_p_wind'],
        precip_2['z223_wind'], precip_2['z258_wind'], precip_2['tau_f'],
        U_2, T_2, gamma_sat_2, h_s_2, h_r, d2h0_2, d18o0_2,
        d_d2h0_d_lat_2, d_d18o0_d_lat_2, is_fit
    )
    
    d2h_grid_2 = iso_2['d2H_grid']
    d18o_grid_2 = iso_2['d18O_grid']
    tau_f_2 = precip_2['tau_f']
    
    # 样本预测 2
    d2h_pred_2 = np.zeros(n_samples)
    d18o_pred_2 = np.zeros(n_samples)
    p_sum_pred_2 = np.zeros(n_samples)
    
    for k in range(n_samples):
        ij = ij_catch[ptr_catch[k]:ptr_catch[k+1]] if k < len(ptr_catch) - 1 else ij_catch[ptr_catch[k]:]
        p_sum_pred_2[k] = sum(p_grid_2[i, j] for i, j in ij)
        if p_sum_pred_2[k] > 0:
            wt = np.array([p_grid_2[i, j] / p_sum_pred_2[k] for i, j in ij])
            d2h_pred_2[k] = sum(wt[m] * d2h_grid_2[i, j] for m, (i, j) in enumerate(ij))
            d18o_pred_2[k] = sum(wt[m] * d18o_grid_2[i, j] for m, (i, j) in enumerate(ij))
    
    # ============ 组合两个风场 ============
    p_grid = p_grid_1 + p_grid_2
    with np.errstate(divide='ignore', invalid='ignore'):
        d2h_grid = np.where(p_grid > 0, (p_grid_1 * d2h_grid_1 + p_grid_2 * d2h_grid_2) / p_grid, np.nan)
        d18o_grid = np.where(p_grid > 0, (p_grid_1 * d18o_grid_1 + p_grid_2 * d18o_grid_2) / p_grid, np.nan)
    
    p_sum_pred = p_sum_pred_1 + p_sum_pred_2
    i_wet = p_sum_pred > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        d2h_pred = np.where(i_wet, (p_sum_pred_1 * d2h_pred_1 + p_sum_pred_2 * d2h_pred_2) / p_sum_pred, np.nan)
        d18o_pred = np.where(i_wet, (p_sum_pred_1 * d18o_pred_1 + p_sum_pred_2 * d18o_pred_2) / p_sum_pred, np.nan)
    
    # 计算统计量
    n_wet = np.sum(i_wet)
    nu = n_wet - n_parameters_free
    if nu > 0 and n_wet > 0:
        r = np.vstack([sample_d2h[i_wet] - d2h_pred[i_wet], sample_d18o[i_wet] - d18o_pred[i_wet]])
        chi_r2 = np.sum(r**2) / nu
        std_residuals = np.sqrt(np.sum(r**2, axis=0))
    else:
        chi_r2 = np.nan
        std_residuals = np.full(n_samples, np.nan)
    
    return {
        'chi_r2': chi_r2, 'nu': nu, 'std_residuals': std_residuals,
        'p_grid_1': p_grid_1, 'd2h_grid_1': d2h_grid_1, 'd18o_grid_1': d18o_grid_1,
        'p_grid_2': p_grid_2, 'd2h_grid_2': d2h_grid_2, 'd18o_grid_2': d18o_grid_2,
        'p_grid': p_grid, 'd2h_grid': d2h_grid, 'd18o_grid': d18o_grid,
        'p_sum_pred_1': p_sum_pred_1, 'd2h_pred_1': d2h_pred_1, 'd18o_pred_1': d18o_pred_1,
        'p_sum_pred_2': p_sum_pred_2, 'd2h_pred_2': d2h_pred_2, 'd18o_pred_2': d18o_pred_2,
        'p_sum_pred': p_sum_pred, 'd2h_pred': d2h_pred, 'd18o_pred': d18o_pred,
        'tau_f_1': tau_f_1, 'tau_f_2': tau_f_2
    }


def main():
    """主函数"""
    print("="*70)
    print("Run033 Replica - MATLAB opiCalc_TwoWinds Replica")
    print("="*70)
    
    start_time = datetime.now()
    
    # 配置
    config = parse_run_file('../OPI_matlab/runs/run033.run')
    
    # 查找数据文件
    search_paths = [
        Path(__file__).parent.parent / 'data',
        Path(__file__).parent.parent.parent / 'OPI_matlab' / 'data',
        Path(__file__).parent.parent / 'examples' / 'data',
        Path.cwd() / 'data',
    ]
    
    data_path = None
    for path in search_paths:
        if path.exists():
            # 检查是否包含所需文件
            sample_path = path / config['sample_file']
            topo_path = path / config['topo_file']
            if sample_path.exists():
                data_path = path
                print(f"Found data directory: {data_path}")
                break
    
    if data_path is None:
        print("\nERROR: Could not find data directory!")
        print("Please ensure the following files are available:")
        print(f"  1. {config['topo_file']}")
        print(f"  2. {config['sample_file']}")
        print("\nSearch paths checked:")
        for p in search_paths:
            print(f"  - {p}")
        print("\nNote: Large .mat files may be excluded by .gitignore")
        return
    
    # 检查地形文件是否存在
    topo_path = data_path / config['topo_file']
    if not topo_path.exists():
        print(f"\nERROR: Topography file not found: {topo_path}")
        print("Please provide the Himalaya topography file.")
        return
    
    # 加载输入数据
    input_data = get_input(str(data_path), config['topo_file'], config['r_tukey'], 
                          config['sample_file'], 28.3)
    
    lon, lat, x, y, h_grid = input_data['lon'], input_data['lat'], input_data['x'], input_data['y'], input_data['h_grid']
    lon0, lat0 = input_data['lon0'], input_data['lat0']
    
    print(f"\nGrid: {h_grid.shape}")
    print(f"Origin: ({lon0:.3f}°E, {lat0:.3f}°N)")
    print(f"Elevation range: {h_grid.min():.0f} ~ {h_grid.max():.0f} m")
    
    # 计算流域节点
    sample_x_pri = input_data['sample_x_pri']
    sample_y_pri = input_data['sample_y_pri']
    sample_lc_pri = input_data['sample_lc_pri']
    
    print("\nComputing catchment nodes...")
    ij_catch, ptr_catch = catchment_nodes(x, y, sample_x_pri, sample_y_pri, sample_lc_pri, h_grid)
    print(f"  Total catchment nodes: {len(ij_catch)}")
    print(f"  Number of samples: {len(ptr_catch) - 1}")
    
    # 准备参数
    beta = config['beta']
    lB, uB = config['lB'], config['uB']
    n_parameters_free = np.sum(lB != uB)
    
    print(f"\nParameters (free: {n_parameters_free}):")
    print(f"  Wind 1: U={beta[0]:.1f} m/s, Az={beta[1]:.1f}°, T0={beta[2]:.1f}K, M={beta[3]:.3f}")
    print(f"  Wind 2: U={beta[10]:.1f} m/s, Az={beta[11]:.1f}°, T0={beta[12]:.1f}K, M={beta[13]:.3f}")
    print(f"  Fraction: {beta[9]:.3f}")
    
    # 执行计算
    print("\n" + "="*70)
    print("Running OPI calculation...")
    print("="*70)
    
    result = calc_two_winds(
        beta, input_data['f_c'], 540.0, x, y, lat, lat0, h_grid, input_data['b_mwl'],
        ij_catch, ptr_catch, input_data['sample_d2h_pri'], input_data['sample_d18o_pri'],
        input_data['cov'], n_parameters_free, False
    )
    
    # 输出结果
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"chiR2 = {result['chi_r2']:.4f}")
    print(f"nu = {result['nu']}")
    print(f"p_grid max = {result['p_grid'].max():.6f}")
    print(f"d2H range = {np.nanmin(result['d2h_grid'])*1000:.1f} ~ {np.nanmax(result['d2h_grid'])*1000:.1f} permil")
    print(f"d18O range = {np.nanmin(result['d18o_grid'])*1000:.1f} ~ {np.nanmax(result['d18o_grid'])*1000:.1f} permil")
    
    # 保存结果
    output_dir = opi_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'run033_replica_{timestamp}.npz'
    
    np.savez(output_file,
        lon=lon, lat=lat, h_grid=h_grid,
        beta=beta, chi_r2=result['chi_r2'], nu=result['nu'],
        p_grid=result['p_grid'], d2h_grid=result['d2h_grid'], d18o_grid=result['d18o_grid'],
        d2h_pred=result['d2h_pred'], d18o_pred=result['d18o_pred']
    )
    
    print(f"\nSaved to: {output_file}")
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"Duration: {duration/60:.2f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()
