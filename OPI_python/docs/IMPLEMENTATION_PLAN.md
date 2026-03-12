# OPI Python 复刻实施计划

## 当前状态速览

```
已完成: 基础工具模块 (30%)
进行中: 核心物理引擎 (0%)
待开始: 应用层、可视化、I/O
```

---

## Phase 1: 核心物理引擎 (优先执行)

### Week 1: 傅里叶地形解

**目标:** 实现 `fourier_solution.py`，完成地形频域转换

**具体任务:**
- [ ] 实现 `wind_grid` 函数（风向坐标系转换）
- [ ] 实现 `fourier_solution` 主函数
- [ ] 实现垂直波数 `kZ` 计算（处理奇点）
- [ ] 编写单元测试，对比 MATLAB 输出

**关键 MATLAB 参考:**
```matlab
% fourierSolution.m 关键逻辑
% 1. 调用 windGrid 转换坐标
% 2. FFT2: hHat = fft2(hWind) / (nS * nT)
% 3. 计算 kZ = sqrt(...) 处理科氏力奇点
% 4. 输出 s, t, kS, kT, hHat, kZ
```

**验收标准:**
- 与 MATLAB 的 `fourierSolution` 输出差异 < 1e-10
- 处理各种网格尺寸和风向

---

### Week 2: 降水网格计算

**目标:** 重构 `precipitation_grid.py`，实现完整 LTOP 算法

**具体任务:**
- [ ] 实现参考降水率计算
- [ ] 实现水汽平衡和蒸发再循环
- [ ] 实现格林函数卷积
- [ ] 实现凝结高度计算

**关键 MATLAB 参考:**
```matlab
% precipitationGrid.m 关键逻辑
% 1. 调用 fourierSolution 获取频域解
% 2. 计算参考降水: P0 = ...
% 3. 水汽比 SHat = ... (含蒸发再循环)
% 4. IFFT2 转回空间域
% 5. 计算 z223, z258 (等温线高度)
```

**验收标准:**
- 降水率网格与 MATLAB 差异 < 1%
- 水汽比和相对湿度计算正确

---

### Week 3: 分馏模型

**目标:** 实现 `fractionation_hydrogen.py` 和 `fractionation_oxygen.py`

**具体任务:**
- [ ] 实现水-汽平衡分馏
- [ ] 实现冰-汽平衡分馏
- [ ] 实现动力学分馏
- [ ] 实现 WBF 区插值

**关键 MATLAB 参考:**
```matlab
% fractionationHydrogen.m
% 1. 温度范围判断 (<248K, 248-268K, >268K)
% 2. 平衡分馏系数 alphaEq
% 3. 动力学分馏系数 alphaKin
% 4. WBF 区线性插值
```

**验收标准:**
- 分馏系数曲线与 MATLAB 一致
- 处理温度边界条件正确

---

### Week 4: 同位素网格计算

**目标:** 重构 `isotope_grid.py`

**具体任务:**
- [ ] 实现垂直平均分馏因子
- [ ] 实现沿风向积分（瑞利蒸馏）
- [ ] 实现蒸发再循环效应
- [ ] 实现纬度梯度校正

**关键 MATLAB 参考:**
```matlab
% isotopeGrid.m 关键逻辑
% 1. 计算分馏因子数组
% 2. 沿风向积分: d2HGrid = ...
% 3. 蒸发再循环校正
% 4. 纬度梯度: d_d2H0_d_lat * (lat - lat0)
```

**验收标准:**
- 同位素网格与 MATLAB 差异 < 0.1‰
- 处理边界条件（网格边缘）正确

---

### Week 5-6: CRS3 优化算法

**目标:** 实现 `fmin_crs3.py`，替换现有的差分进化

**具体任务:**
- [ ] 研究 CRS3 算法论文 (Price 1987, Brachetti 1997)
- [ ] 实现基本 CRS3
- [ ] 实现改进：加权质心、二次近似
- [ ] 实现并行支持
- [ ] 实现重启功能

**验收标准:**
- 找到的最优解与 MATLAB 的 CRS3 一致
- 收敛速度相当

---

## Phase 2: 应用层完善

### Week 7: 单风场主程序

**目标:** 完善 `opi_calc_one_wind.py`

**具体任务:**
- [ ] 连接 `calc_one_wind` 真实计算
- [ ] 实现完整结果保存
- [ ] 实现拟合统计量计算
- [ ] 与 MATLAB `opiCalc_OneWind` 输出对比

---

### Week 8: 输入数据加载

**目标:** 实现 `get_input.py`

**具体任务:**
- [ ] 实现 MAT 文件读取（支持 v7.3）
- [ ] 实现 Excel 样本文件读取
- [ ] 实现 Tukey 窗口平滑
- [ ] 实现 MWL 拟合

---

### Week 9: 参数拟合

**目标:** 完善 `opi_fit_one_wind.py`

**具体任务:**
- [ ] 连接 CRS3 优化器
- [ ] 实现约束处理
- [ ] 实现结果保存到运行文件
- [ ] 实现进度输出

---

### Week 10: 双风场功能

**目标:** 完善 `opi_calc_two_wind.py` 和 `opi_fit_two_winds.py`

**具体任务:**
- [ ] 实现 `calc_two_wind`
- [ ] 实现大陆分水岭处理
- [ ] 实现结果加权合并
- [ ] 实现双风场拟合

---

## Phase 3: 可视化和 I/O

### Week 11-12: 图表和地图

**目标:** 完善可视化功能

**具体任务:**
- [ ] 完善 `opi_plots_one_wind.py`（7个图表）
- [ ] 实现 `opi_maps_one_wind.py`（13个地图）
- [ ] 实现 `opi_predict_calc.py`
- [ ] 实现 `opi_predict_plot.py`

---

## 详细任务分解

### 任务 1.1: fourier_solution.py

```python
def fourier_solution(h_grid, x, y, U, azimuth, NM, f_c, h_rho):
    """
    计算地形在风向坐标系中的傅里叶解
    
    Parameters
    ----------
    h_grid : ndarray (nY, nX)
        地形高程网格
    x, y : ndarray
        网格坐标（米）
    U : float
        风速 (m/s)
    azimuth : float
        风向角（度，从北顺时针）
    NM : float
        浮力频率 (rad/s)
    f_c : float
        科氏参数 (rad/s)
    h_rho : float
        密度标高 (m)
    
    Returns
    -------
    dict : 包含 s, t, Sxy, Txy, kS, kT, hHat, kZ, hWind
    """
    # 1. 调用 wind_grid 创建风向坐标系
    # 2. 插值地形到风向网格
    # 3. FFT2 计算 hHat
    # 4. 计算波数 kS, kT
    # 5. 计算垂直波数 kZ（处理科氏力奇点）
    # 6. 返回所有变量
    pass
```

**测试要求:**
```python
def test_fourier_solution():
    # 加载 MATLAB 保存的测试数据
    mat_data = loadmat('test_data/fourier_solution_test.mat')
    
    result = fourier_solution(
        mat_data['h_grid'], 
        mat_data['x'], 
        mat_data['y'],
        mat_data['U'][0,0],
        mat_data['azimuth'][0,0],
        mat_data['NM'][0,0],
        mat_data['f_c'][0,0],
        mat_data['h_rho'][0,0]
    )
    
    # 对比每个输出变量
    assert_allclose(result['hHat'], mat_data['hHat'], rtol=1e-10)
    assert_allclose(result['kZ'], mat_data['kZ'], rtol=1e-10)
    # ... 其他变量
```

---

### 任务 1.2: precipitation_grid.py

```python
def precipitation_grid(x, y, h_grid, U, azimuth, NM, f_c, kappa, tau_c, 
                       h_rho, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
                       rho_s0, h_s, f_p0):
    """
    计算降水率及相关变量的空间分布
    
    核心算法：LTOP (Linear Theory of Orographic Precipitation)
    """
    # 1. 调用 fourier_solution 获取频域解
    # 2. 计算参考降水率 P0
    # 3. 计算格林函数 GS, GC, GF
    # 4. 计算水汽比 SHat（含蒸发再循环）
    # 5. IFFT2 转回空间域得到 Sxy, Txy
    # 6. 计算降水率 p_grid
    # 7. 计算凝结高度 z223, z258
    # 8. 计算下落时间 tau_f
    pass
```

---

### 任务 1.3: fractionation_hydrogen.py

```python
def fractionation_hydrogen(T):
    """
    计算氢同位素分馏因子
    
    基于 Ciais & Jouzel (1994) MCIM 模型
    
    Parameters
    ----------
    T : float or ndarray
        温度 (K)
    
    Returns
    -------
    alpha : float or ndarray
        分馏因子 (R_precip / R_vapor)
    """
    # 1. 判断温度范围
    # 2. 计算水-汽平衡分馏 (T > 273.15)
    # 3. 计算冰-汽平衡分馏 (T < 253.15)
    # 4. WBF 区 (253.15 <= T <= 273.15) 线性插值
    # 5. 应用动力学效应
    pass
```

---

## 开发工具和脚本

### 创建测试数据提取脚本

建议创建一个 MATLAB 脚本，用于提取测试数据：

```matlab
% extract_test_data.m
% 运行示例案例并保存中间结果

% 运行 opiCalc_OneWind 的简化版本
[runPath, runName] = getRunFile();
runData = getRunFile(fullfile(runPath, runName));
inputData = getInput(...);

% 保存中间结果
save('test_data/base_state_output.mat', 'z_bar', 'T', 'gamma_env', ...);
save('test_data/fourier_solution_output.mat', 's', 't', 'hHat', 'kZ', ...);
save('test_data/precipitation_output.mat', 'p_grid', 'Sxy', 'Txy', ...);
save('test_data/isotope_output.mat', 'd2H_grid', 'd18O_grid', ...);
```

---

## 代码审查清单

每个模块实现后，检查以下事项：

- [ ] 算法与 MATLAB 代码一致
- [ ] 所有边界条件处理正确
- [ ] 数值精度满足要求（与 MATLAB 对比）
- [ ] 单元测试覆盖主要分支
- [ ] 文档字符串完整
- [ ] 异常处理完善

---

## 下一步行动

1. **立即开始**: 创建测试数据提取 MATLAB 脚本
2. **本周内**: 开始实现 `fourier_solution.py`
3. **持续**: 与 MATLAB 输出进行数值对比

**建议的开发环境:**
- 双屏显示 MATLAB 和 Python 代码
- 准备 MATLAB 测试数据
- 使用 PyCharm 或 VSCode 调试
