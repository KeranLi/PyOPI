#!/usr/bin/env python
"""
测试OPI Python代码的降水计算，检查是否符合物理规律

物理规律预期：
1. 迎风坡前：雨影效应（低降水）
2. 迎风坡：降水最大值
3. 背风坡后：第二个降水带
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_precipitation_calculation():
    """测试降水计算并可视化结果"""
    
    print("="*70)
    print("测试OPI Python降水计算")
    print("="*70)
    
    # 导入必要的模块
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OPI_python'))
    
    from opi.physics.fourier import fourier_solution
    from opi.physics.precipitation import precipitation_grid, isotherm
    from opi.physics.thermodynamics import base_state
    
    # 创建一个简单的东西向剖面地形（2D山脉）
    # 风向90°（向东），这样我们可以清楚地看到迎风坡和背风坡效应
    
    # 网格设置
    nx, ny = 200, 100
    x = np.linspace(-200e3, 200e3, nx)  # -200km to 200km
    y = np.linspace(-100e3, 100e3, ny)  # -100km to 100km
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    print(f"\n网格设置:")
    print(f"  x范围: {x.min()/1000:.0f} km to {x.max()/1000:.0f} km, dx={dx/1000:.1f} km")
    print(f"  y范围: {y.min()/1000:.0f} km to {y.max()/1000:.0f} km, dy={dy/1000:.1f} km")
    
    # 创建高斯山脉
    X, Y = np.meshgrid(x, y)
    mountain_height = 3000  # 3km山峰
    sigma_x = 30e3  # 30km半宽
    sigma_y = 50e3  # 50km半宽
    
    h_grid = mountain_height * np.exp(-(X**2)/(2*sigma_x**2) - (Y**2)/(2*sigma_y**2))
    
    print(f"\n地形参数:")
    print(f"  山峰高度: {mountain_height} m")
    print(f"  x方向半宽: {sigma_x/1000:.0f} km")
    print(f"  y方向半宽: {sigma_y/1000:.0f} km")
    
    # 物理参数
    U = 10.0          # 风速 m/s
    azimuth = 90.0    # 风向：90°=向东（从西向东吹）
    T0 = 290.0        # 海平面温度 K
    M = 0.25          # 山高度数
    kappa = 0.0       # 无涡流扩散
    tau_c = 1000.0    # 凝结时间 s
    f_c = 1e-4        # 科里奥利参数
    f_p0 = 1.0        # 无蒸发
    
    print(f"\n物理参数:")
    print(f"  风速: {U} m/s")
    print(f"  风向: {azimuth}° (向东)")
    print(f"  温度: {T0} K")
    print(f"  山高度数 M: {M}")
    print(f"  凝结时间: {tau_c} s")
    
    # 计算NM
    h_max = h_grid.max()
    NM = M * U / h_max
    
    print(f"\n计算参数:")
    print(f"  最大地形高度: {h_max} m")
    print(f"  浮力频率 NM: {NM:.6f} rad/s")
    
    # 计算基础状态
    print("\n[1] 计算大气基础状态...")
    z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
    print(f"  水汽密度 rho_s0: {rho_s0:.6f} kg/m³")
    print(f"  水汽标高 h_s: {h_s:.0f} m")
    print(f"  密度标高 h_rho: {h_rho:.0f} m")
    print(f"  gamma_ratio: {gamma_ratio:.4f}")
    
    # 计算降水
    print("\n[2] 计算降水分布...")
    result = precipitation_grid(
        x, y, h_grid, U, azimuth, NM, f_c, kappa, tau_c,
        h_rho, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
        rho_s0, h_s, f_p0
    )
    
    p_grid = result['p_grid']
    s = result['s']
    t = result['t']
    Sxy = result['Sxy']
    Txy = result['Txy']
    h_wind = result['h_wind']
    tau_f = result['tau_f']
    
    print(f"\n降水结果:")
    print(f"  降水率范围: {p_grid.min()*1000*86400:.4f} to {p_grid.max()*1000*86400:.4f} mm/day")
    print(f"  平均降水率: {p_grid.mean()*1000*86400:.4f} mm/day")
    print(f"  下落时间 tau_f: {tau_f:.1f} s")
    print(f"  s向量范围: {s.min()/1000:.1f} to {s.max()/1000:.1f} km")
    print(f"  t向量范围: {t.min()/1000:.1f} to {t.max()/1000:.1f} km")
    print(f"  h_wind形状: {h_wind.shape}")
    print(f"  p_grid形状: {p_grid.shape}")
    
    # 分析沿风向的降水剖面
    print("\n[3] 分析沿风向(x方向)的降水剖面...")
    
    # 找到中心y索引
    center_y_idx = ny // 2
    precip_cross_section = p_grid[center_y_idx, :]
    topo_cross_section = h_grid[center_y_idx, :]
    
    # 找到地形最大值位置
    peak_idx = np.argmax(topo_cross_section)
    print(f"  山峰位置索引: {peak_idx}, x={x[peak_idx]/1000:.1f} km")
    
    # 分析降水分布
    # 迎风坡前（山峰左侧）
    upwind_start = max(0, peak_idx - 50)
    upwind_end = peak_idx
    precip_upwind = precip_cross_section[upwind_start:upwind_end].mean()
    
    # 迎风坡
    windward_start = peak_idx - 20 if peak_idx > 20 else 0
    windward_end = peak_idx
    precip_windward = precip_cross_section[windward_start:windward_end].mean()
    
    # 背风坡后
    lee_start = peak_idx + 10
    lee_end = min(nx, peak_idx + 50)
    precip_lee = precip_cross_section[lee_start:lee_end].mean()
    
    print(f"\n  各区域平均降水率 (mm/day):")
    print(f"    迎风坡前 (x<{x[upwind_end]/1000:.0f}km): {precip_upwind*1000*86400:.4f}")
    print(f"    迎风坡 ({x[windward_start]/1000:.0f}km<x<{x[windward_end]/1000:.0f}km): {precip_windward*1000*86400:.4f}")
    print(f"    背风坡后 (x>{x[lee_start]/1000:.0f}km): {precip_lee*1000*86400:.4f}")
    
    # 物理规律检查
    print("\n[4] 物理规律检查:")
    is_valid = True
    
    # 检查1：迎风坡应该比迎风坡前降水多
    if precip_windward > precip_upwind:
        print("  ✓ 迎风坡降水 > 迎风坡前降水 (符合预期)")
    else:
        print("  ✗ 迎风坡降水 <= 迎风坡前降水 (不符合预期!)")
        is_valid = False
    
    # 检查2：背风坡应该有降水（可能有第二个带）
    if precip_lee > precip_upwind * 0.5:  # 至少达到迎风坡前的一半
        print("  ✓ 背风坡有显著降水 (可能有第二个降水带)")
    else:
        print("  ! 背风坡降水较少")
    
    # 检查3：山顶附近应该有最大降水
    peak_region = precip_cross_section[peak_idx-10:peak_idx+10].mean()
    if peak_region > precip_upwind:
        print("  ✓ 山峰附近降水 > 迎风坡前 (符合地形抬升效应)")
    else:
        print("  ✗ 山峰附近降水 <= 迎风坡前 (不符合地形抬升效应!)")
        is_valid = False
    
    # 可视化
    print("\n[5] 创建可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1：地形
    ax = axes[0, 0]
    im = ax.contourf(X/1000, Y/1000, h_grid, levels=20, cmap='terrain')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title('Topography (m)')
    plt.colorbar(im, ax=ax)
    # 添加风向箭头
    ax.annotate('', xy=(150, -80), xytext=(100, -80),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(125, -90, 'Wind', ha='center', color='red', fontsize=10)
    
    # 图2：降水分布
    ax = axes[0, 1]
    precip_mm_day = p_grid * 1000 * 86400
    im = ax.contourf(X/1000, Y/1000, precip_mm_day, levels=20, cmap='Blues')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title('Precipitation Rate (mm/day)')
    plt.colorbar(im, ax=ax)
    # 添加风向箭头
    ax.annotate('', xy=(150, -80), xytext=(100, -80),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(125, -90, 'Wind', ha='center', color='red', fontsize=10)
    
    # 图3：沿风向剖面
    ax = axes[1, 0]
    ax2 = ax.twinx()
    
    # 降水
    line1 = ax.plot(x/1000, precip_mm_day[center_y_idx, :], 'b-', linewidth=2, label='Precipitation')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('Precipitation (mm/day)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.axvline(x[peak_idx]/1000, color='g', linestyle='--', alpha=0.5, label='Mountain Peak')
    ax.axvspan(x[peak_idx]/1000 - 20, x[peak_idx]/1000, alpha=0.2, color='green', label='Windward')
    ax.axvspan(x[peak_idx]/1000, x[peak_idx]/1000 + 40, alpha=0.2, color='orange', label='Lee')
    
    # 地形
    line2 = ax2.plot(x/1000, topo_cross_section, 'k-', linewidth=1.5, label='Topography')
    ax2.set_ylabel('Elevation (m)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.fill_between(x/1000, topo_cross_section, alpha=0.3, color='gray')
    
    ax.set_title('Cross-section along wind direction (center y)')
    ax.legend(loc='upper left')
    
    # 图4：沿垂直风向剖面
    ax = axes[1, 1]
    center_x_idx = nx // 2
    ax2 = ax.twinx()
    
    # 降水
    ax.plot(y/1000, precip_mm_day[:, center_x_idx], 'b-', linewidth=2)
    ax.set_xlabel('y (km)')
    ax.set_ylabel('Precipitation (mm/day)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # 地形
    ax2.plot(y/1000, h_grid[:, center_x_idx], 'k-', linewidth=1.5)
    ax2.set_ylabel('Elevation (m)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.fill_between(y/1000, h_grid[:, center_x_idx], alpha=0.3, color='gray')
    
    ax.set_title('Cross-section across wind direction (center x)')
    
    plt.tight_layout()
    plt.savefig('test_precipitation_result.png', dpi=150, bbox_inches='tight')
    print("  保存图片: test_precipitation_result.png")
    
    # 额外分析：检查s,t坐标系
    print("\n[6] 坐标系分析:")
    print(f"  Sxy形状: {Sxy.shape}")
    print(f"  Sxy范围: {Sxy.min()/1000:.1f} to {Sxy.max()/1000:.1f} km")
    print(f"  Txy范围: {Txy.min()/1000:.1f} to {Txy.max()/1000:.1f} km")
    
    # 检查s方向是否沿x方向（因为风向是90°）
    s_at_y0 = Sxy[center_y_idx, :]
    x_check = X[center_y_idx, :]
    if np.allclose(s_at_y0, x_check * np.sin(np.deg2rad(azimuth)) + Y[center_y_idx, :].mean() * np.cos(np.deg2rad(azimuth)), rtol=0.01):
        print("  ✓ Sxy沿风向正确")
    else:
        print("  ! Sxy可能与风向不一致")
    
    print("\n" + "="*70)
    if is_valid:
        print("测试结论: 降水分布符合物理规律")
    else:
        print("测试结论: 降水分布可能有问题!")
    print("="*70)
    
    plt.show()
    
    return result, is_valid


if __name__ == "__main__":
    result, is_valid = test_precipitation_calculation()
