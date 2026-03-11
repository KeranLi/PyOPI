"""
Run033 Reproduction Script
一比一复刻 MATLAB OPI 3.7 双风场模拟
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opi.get_input import get_input, grid_read, tukey_window, estimate_mwl
from opi.coordinates import lonlat2xy
from opi.calc_two_winds import calc_two_winds
from opi.catchment_nodes import catchment_nodes
from opi.constants import RADIUS_EARTH


def parse_run033():
    """
    解析 run033.run 文件中的参数
    """
    # 直接硬编码 run033 的参数（从 .run 文件解析）
    config = {
        'run_title': 'run 33, Himalaya area, OPI 3.7, two wind, rTukey = 0.5, M<1, free fP, mu = 20',
        'parallel_option': 0,
        'data_path': r'F:\code\OPI-Orographic-Precipitation-and-Isotopes\OPI_matlab\data',
        'topo_file': 'Himalaya topography_April 2023_Gebco1kmMakima.mat',
        'r_tukey': 0.5,
        'sample_file': 'Himalaya Water Isotopes 8 March 2023.xlsx',
        'cont_divide_file': 'Asia_contDivideLonLat.mat',
        'map_limits': [77.5, 97.5, 20.0, 36.0],  # [minLon, maxLon, minLat, maxLat]
        'section_origin': [90.26876, 29.19308],  # [lon, lat]
        'crs_params': [20, 1e-4],  # [mu, epsilon0]
        'restart_file': 'no',
        'parameter_labels': [
            'U', 'Azimuth', 'T0', 'M', 'kappa', 'tauC', 'd2H0', 'dLat', 'fP',
            'fraction',
            'U2', 'Az2', 'T02', 'M2', 'kappa2', 'tauC2', 'd2H0_2', 'dLat2', 'fP2'
        ],
        'exponents': [0] * 19,
        'lB': [0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0, 0,
               0.1, 0, 270, 0, 0, 0, -0.060, -0.015, 0],
        'uB': [20, 90, 300, 1, 1000000, 2500, 0, 0, 1, 1,
               20, 180, 300, 1, 1000000, 2500, 0, 0, 1],
        # 最优解 beta 值（从 run 文件最后一行）
        'beta': [
            6.6, 33.8, 293.5, 0.340, 233368, 1296, -0.0026, -0.000533, 1.0, 0.52,
            16.0, 125.7, 294.0, 0.693, 722120, 2450, -0.0001, -0.000007, 1.0
        ]
    }
    return config


def create_synthetic_himalaya_data(config):
    """
    创建合成喜马拉雅地形数据（用于测试）
    由于缺少真实 .mat 文件，使用合成数据
    """
    print("创建合成喜马拉雅地形数据...")
    
    # 从 map_limits 创建网格
    min_lon, max_lon, min_lat, max_lat = config['map_limits']
    
    # 模拟 1km 分辨率
    d_lon = 0.01  # ~1km
    d_lat = 0.01
    
    lon = np.arange(min_lon, max_lon + d_lon, d_lon)
    lat = np.arange(min_lat, max_lat + d_lat, d_lat)
    
    # 创建网格
    LON, LAT = np.meshgrid(lon, lat)
    
    # 创建合成喜马拉雅地形（高斯山脉）
    # 主峰在青藏高原区域
    peak_lon = 87.0
    peak_lat = 28.0
    h_grid = 8000 * np.exp(-((LON - peak_lon)**2 / 2 + (LAT - peak_lat)**2 / 2) / 4)
    
    # 添加次要山脉
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
    
    print(f"  网格大小: {h_grid.shape}")
    print(f"  经度范围: {lon.min():.2f} ~ {lon.max():.2f}")
    print(f"  纬度范围: {lat.min():.2f} ~ {lat.max():.2f}")
    print(f"  高程范围: {h_grid.min():.0f} ~ {h_grid.max():.0f} m")
    
    return lon, lat, h_grid


def load_samples(config, lon, lat):
    """
    加载样本数据
    """
    data_path = config['data_path']
    sample_file = config['sample_file']
    
    # 尝试多种路径
    possible_paths = [
        os.path.join(data_path, sample_file),
        os.path.join('OPI_matlab/data', sample_file),
        os.path.join('../OPI_matlab/data', sample_file),
        sample_file
    ]
    
    sample_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sample_path = path
            break
    
    if sample_path is None:
        print(f"警告: 找不到样本文件，创建合成样本")
        return create_synthetic_samples(config)
    
    print(f"加载样本数据: {sample_path}")
    
    # 读取 Excel 文件
    try:
        df = pd.read_excel(sample_path, header=None, comment='%')
    except Exception as e:
        print(f"  读取 Excel 失败: {e}")
        return create_synthetic_samples(config)
    
    # 过滤注释行
    df = df[~df.iloc[:, 0].astype(str).str.strip().str.startswith('%')]
    df = df.dropna()
    
    # 解析列（根据 run033 的实际格式）
    # 列顺序: Longitude, Latitude, Elevation, d2H, d18O, Type
    sample_line = df.index.values + 1
    sample_lon = df.iloc[:, 0].values
    sample_lat = df.iloc[:, 1].values
    # sample_elev = df.iloc[:, 2].values  # 海拔，暂时不用
    sample_d2h = df.iloc[:, 3].values * 1e-3  # 转小数
    sample_d18o = df.iloc[:, 4].values * 1e-3
    sample_lc = df.iloc[:, 5].astype(str).str.upper().str[0].values
    
    print(f"  样本数量: {len(sample_lon)}")
    print(f"  Local (L): {np.sum(sample_lc == 'L')}")
    print(f"  Catchment (C): {np.sum(sample_lc == 'C')}")
    
    return {
        'sample_line': sample_line,
        'sample_lon': sample_lon,
        'sample_lat': sample_lat,
        'sample_d2h': sample_d2h,
        'sample_d18o': sample_d18o,
        'sample_lc': sample_lc
    }


def create_synthetic_samples(config):
    """
    创建合成样本数据
    """
    print("创建合成样本数据...")
    
    # 在区域内创建一些样本点
    min_lon, max_lon, min_lat, max_lat = config['map_limits']
    
    np.random.seed(42)
    n_samples = 20
    
    sample_lon = np.random.uniform(min_lon + 2, max_lon - 2, n_samples)
    sample_lat = np.random.uniform(min_lat + 2, max_lat - 2, n_samples)
    sample_d2h = np.random.uniform(-150, -50, n_samples) * 1e-3
    sample_d18o = sample_d2h / 8 + np.random.normal(0, 5e-3, n_samples)
    sample_lc = np.random.choice(['L', 'C'], n_samples)
    
    return {
        'sample_line': np.arange(1, n_samples + 1),
        'sample_lon': sample_lon,
        'sample_lat': sample_lat,
        'sample_d2h': sample_d2h,
        'sample_d18o': sample_d18o,
        'sample_lc': sample_lc
    }


def run_two_winds_simulation(config, lon, lat, h_grid, samples):
    """
    运行双风场模拟
    """
    print("\n" + "="*60)
    print("运行双风场 OPI 模拟")
    print("="*60)
    
    # 解析 beta 参数
    beta = np.array(config['beta'])
    
    # 提取参数
    # 风场 1
    U_1 = beta[0]
    azimuth_1 = beta[1]
    T0_1 = beta[2]
    M_1 = beta[3]
    kappa_1 = beta[4]
    tau_c_1 = beta[5]
    d2h0_1 = beta[6]
    d_d2h0_d_lat_1 = beta[7]
    f_p0_1 = beta[8]
    fraction = beta[9]
    
    # 风场 2
    U_2 = beta[10]
    azimuth_2 = beta[11]
    T0_2 = beta[12]
    M_2 = beta[13]
    kappa_2 = beta[14]
    tau_c_2 = beta[15]
    d2h0_2 = beta[16]
    d_d2h0_d_lat_2 = beta[17]
    f_p0_2 = beta[18]
    
    print("\n参数设置:")
    print(f"  风场 1 (fraction={fraction:.2f}):")
    print(f"    U={U_1:.1f} m/s, Azimuth={azimuth_1:.1f} deg")
    print(f"    T0={T0_1:.1f} K, M={M_1:.3f}")
    print(f"    kappa={kappa_1:.0f} m^2/s, tau_c={tau_c_1:.0f} s")
    print(f"    d2H0={d2h0_1*1000:.1f} permil")
    
    print(f"  风场 2 (fraction={1-fraction:.2f}):")
    print(f"    U={U_2:.1f} m/s, Azimuth={azimuth_2:.1f} deg")
    print(f"    T0={T0_2:.1f} K, M={M_2:.3f}")
    print(f"    kappa={kappa_2:.0f} m^2/s, tau_c={tau_c_2:.0f} s")
    print(f"    d2H0={d2h0_2*1000:.1f} permil")
    
    # 计算地图原点
    sample_lon = samples['sample_lon']
    sample_lat = samples['sample_lat']
    
    if len(sample_lon) > 0:
        lon0 = np.mean(sample_lon)
        lat0 = np.mean(sample_lat)
    else:
        lon0 = np.mean(lon)
        lat0 = np.mean(lat)
    
    print(f"\n地图原点: lon={lon0:.4f}, lat={lat0:.4f}")
    
    # 计算 Coriolis 参数
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    print(f"Coriolis 参数: f_c={f_c*1e4:.4f} x 10^-4 rad/s")
    
    # 转换为投影坐标
    x, y = lonlat2xy(lon, lat, lon0, lat0)
    sample_x, sample_y = lonlat2xy(sample_lon, sample_lat, lon0, lat0)
    
    # 计算 MWL
    sample_d2h = samples['sample_d2h']
    sample_d18o = samples['sample_d18o']
    sample_lc = samples['sample_lc']
    
    sd_res_ratio = 28.3
    b_mwl, sd_min, sd_max, cov, i_fit = estimate_mwl(sample_d18o, sample_d2h, sd_res_ratio)
    
    print(f"\n气象水线 (MWL):")
    print(f"  d2H = {b_mwl[0]*1000:.2f} + {b_mwl[1]:.2f} * d18O")
    
    # 分割样本为主要和改变样本
    sample_line = samples['sample_line']
    sample_d2h_alt = sample_d2h[~i_fit]
    sample_d18o_alt = sample_d18o[~i_fit]
    sample_lc_alt = sample_lc[~i_fit]
    sample_line_alt = sample_line[~i_fit]
    
    sample_d2h = sample_d2h[i_fit]
    sample_d18o = sample_d18o[i_fit]
    sample_lc = sample_lc[i_fit]
    sample_line = sample_line[i_fit]
    sample_x = sample_x[i_fit]
    sample_y = sample_y[i_fit]
    
    print(f"  主要样本: {len(sample_d2h)}")
    print(f"  改变样本: {len(sample_d2h_alt)}")
    
    # 计算流域节点
    ij_catch, ptr_catch = catchment_nodes(
        sample_x, sample_y, sample_lc, x, y, h_grid
    )
    
    print(f"\n流域节点: {len(ij_catch)} 个网格点")
    
    # 自由参数数量
    lB = np.array(config['lB'])
    uB = np.array(config['uB'])
    n_parameters_free = np.sum(lB != uB)
    
    print(f"自由参数数量: {n_parameters_free}")
    
    # 运行双风场计算
    print("\n执行双风场计算...")
    h_r = 540.0
    
    try:
        result = calc_two_winds(
            beta=beta,
            f_c=f_c,
            h_r=h_r,
            x=x,
            y=y,
            lat=lat,
            lat0=lat0,
            h_grid=h_grid,
            b_mwl_sample=b_mwl,
            ij_catch=ij_catch,
            ptr_catch=ptr_catch,
            sample_d2h=sample_d2h,
            sample_d18o=sample_d18o,
            cov=cov,
            n_parameters_free=n_parameters_free,
            is_fit=False
        )
        
        print("\n计算完成!")
        print(f"  chiR2: {result[0]:.4f}")
        print(f"  nu: {result[1]}")
        
        # 提取结果
        chiR2, nu, std_residuals = result[0], result[1], result[2]
        
        # 风场 1 结果
        z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1 = result[3:8]
        rho_s0_1, h_s_1, rho0_1, h_rho_1 = result[8:12]
        d18o0_1, d_d18o0_d_lat_1, tau_f_1, p_grid_1 = result[12:16]
        f_m_grid_1, r_h_grid_1, d2h_grid_1, d18o_grid_1 = result[16:20]
        p_sum_pred_1, d2h_pred_1, d18o_pred_1 = result[20:23]
        
        # 风场 2 结果
        z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2 = result[23:28]
        rho_s0_2, h_s_2, rho0_2, h_rho_2 = result[28:32]
        d18o0_2, d_d18o0_d_lat_2, tau_f_2, p_grid_2 = result[32:36]
        f_m_grid_2, r_h_grid_2, d2h_grid_2, d18o_grid_2 = result[36:40]
        p_sum_pred_2, d2h_pred_2, d18o_pred_2 = result[40:43]
        
        # 组合结果
        p_grid, d2h_grid, d18o_grid = result[43:46]
        p_sum_pred, d2h_pred, d18o_pred = result[46:49]
        fraction_p_grid = result[49]
        
        print("\n结果摘要:")
        print(f"  风场 1 降水最大值: {np.nanmax(p_grid_1):.4f}")
        print(f"  风场 2 降水最大值: {np.nanmax(p_grid_2):.4f}")
        print(f"  组合降水最大值: {np.nanmax(p_grid):.4f}")
        print(f"  d2H 范围: {np.nanmin(d2h_grid)*1000:.1f} ~ {np.nanmax(d2h_grid)*1000:.1f} permil")
        print(f"  d18O 范围: {np.nanmin(d18o_grid)*1000:.1f} ~ {np.nanmax(d18o_grid)*1000:.1f} permil")
        
        return {
            'config': config,
            'lon': lon,
            'lat': lat,
            'x': x,
            'y': y,
            'h_grid': h_grid,
            'beta': beta,
            'chiR2': chiR2,
            'nu': nu,
            'p_grid_1': p_grid_1,
            'p_grid_2': p_grid_2,
            'p_grid': p_grid,
            'd2h_grid_1': d2h_grid_1,
            'd2h_grid_2': d2h_grid_2,
            'd2h_grid': d2h_grid,
            'd18o_grid_1': d18o_grid_1,
            'd18o_grid_2': d18o_grid_2,
            'd18o_grid': d18o_grid,
            'd2h_pred': d2h_pred,
            'd18o_pred': d18o_pred,
            'sample_x': sample_x,
            'sample_y': sample_y,
            'sample_d2h': sample_d2h,
            'sample_d18o': sample_d18o,
            'sample_lc': sample_lc
        }
        
    except Exception as e:
        print(f"\n计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results, output_dir='outputs'):
    """
    保存结果
    """
    if results is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'run033_python_{timestamp}.npz')
    
    # 保存关键结果
    np.savez(
        output_file,
        lon=results['lon'],
        lat=results['lat'],
        h_grid=results['h_grid'],
        beta=results['beta'],
        chiR2=results['chiR2'],
        nu=results['nu'],
        p_grid_1=results['p_grid_1'],
        p_grid_2=results['p_grid_2'],
        p_grid=results['p_grid'],
        d2h_grid_1=results['d2h_grid_1'],
        d2h_grid_2=results['d2h_grid_2'],
        d2h_grid=results['d2h_grid'],
        d18o_grid_1=results['d18o_grid_1'],
        d18o_grid_2=results['d18o_grid_2'],
        d18o_grid=results['d18o_grid'],
        d2h_pred=results['d2h_pred'],
        d18o_pred=results['d18o_pred'],
        sample_x=results['sample_x'],
        sample_y=results['sample_y'],
        sample_d2h=results['sample_d2h'],
        sample_d18o=results['sample_d18o']
    )
    
    print(f"\n结果已保存: {output_file}")


def main():
    """
    主函数
    """
    print("="*60)
    print("Run033 复刻 - OPI 3.7 双风场模拟")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 解析配置
    config = parse_run033()
    print(f"\n运行标题: {config['run_title']}")
    
    # 创建合成地形数据
    lon, lat, h_grid = create_synthetic_himalaya_data(config)
    
    # 加载样本数据
    samples = load_samples(config, lon, lat)
    
    # 运行模拟
    results = run_two_winds_simulation(config, lon, lat, h_grid, samples)
    
    # 保存结果
    save_results(results)
    
    print("\n" + "="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    results = main()
