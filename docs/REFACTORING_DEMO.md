# OPI Python 重构演示

这个文档展示了重构前后的代码对比。

## 1. 参数定义

### 重构前 (MATLAB风格)
```python
# 9个浮点数，容易混淆顺序
beta = [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7]

# 使用时需要记住索引
U = beta[0]           # 风速
azimuth = beta[1]     # 方位角
T0 = beta[2]          # 温度
# ... 容易出错
```

### 重构后 (Pythonic风格)
```python
from opi.models import OneWindParameters, WindField, AtmosphereState

# 自描述的参数对象
params = OneWindParameters(
    wind=WindField(speed=10.0, azimuth=90.0),
    atmosphere=AtmosphereState(
        sea_level_temperature=290.0,
        buoyancy_frequency=0.01
    ),
    isotopes=IsotopeParameters(base_d2h=-5e-3),
    microphysics=MicrophysicsParameters(
        eddy_diffusion=0.0,
        condensation_time=1000.0
    )
)

# 自动验证
params.wind.speed  # 清晰！
```

## 2. 计算器使用

### 重构前
```python
# 函数调用，长参数列表
result = calc_one_wind(
    beta, f_c, h_r, x, y, lat, lat0, h_grid,
    b_mwl_sample, ij_catch, ptr_catch,
    sample_d2h, sample_d18o, cov, n_parameters_free, is_fit
)

# 返回元组，容易混淆
chi_r2, nu, std_residuals, z_bar, T, ... = result
```

### 重构后
```python
from opi.calculators import OneWindCalculatorNew

# 创建计算器实例
calculator = OneWindCalculatorNew(config)
calculator.topography = topography

# 计算
results = calculator.calculate(params)

# 访问结果
print(results.precipitation.mean)
print(results.d2h.mean_per_mil)
```

## 3. 错误处理

### 重构前
```python
try:
    result = calc_one_wind(...)
except Exception as e:
    print(f"Error: {e}")
    # 不知道是什么错误
```

### 重构后
```python
from opi.core.exceptions import ValidationError, CalculationError

try:
    results = calculator.calculate(params)
except ValidationError as e:
    print(f"Invalid input: {e.message}")
    print(f"Details: {e.details}")
except CalculationError as e:
    print(f"Calculation failed: {e.message}")
```

## 4. 配置管理

### 重构前
```python
# 分散的魔法数字
NM = 0.01
tau_c = 1000.0
padding = 2.0
```

### 重构后
```python
from opi.models import OPIConfig, SolverConfig

# 结构化配置
config = OPIConfig(
    solver=SolverConfig(
        padding_factor=2.0,
        handle_singularities=True
    ),
    optimizer=OptimizerConfig(
        max_iterations=10000,
        mu=25
    )
)

# 保存和加载
config.save_yaml("opi_config.yaml")
config = OPIConfig.from_yaml("opi_config.yaml")
```

## 5. 代码组织

### 重构前
```
opi/
├── opi_calc_one_wind.py      # 混合了所有逻辑
├── calc_one_wind.py          # 核心计算
├── fourier_solution.py       # FFT
├── precipitation_grid.py     # 降水
└── ...
```

### 重构后
```
opi/
├── models/                   # 数据模型
│   ├── parameters.py        # 参数类
│   ├── results.py           # 结果类
│   ├── domain.py            # 网格/地形
│   └── config.py            # 配置
├── core/                     # 核心抽象
│   ├── base.py              # 基类
│   ├── calculator.py        # 计算器接口
│   └── exceptions.py        # 异常层次
├── solvers/                  # 物理求解器
│   ├── fourier.py           # FFT
│   ├── precipitation.py     # 降水
│   └── isotope.py           # 同位素
├── calculators/              # 应用层
│   ├── one_wind.py          # 单风场
│   └── two_winds.py         # 双风场
└── io/                       # 输入输出
    ├── loaders.py
    └── exporters.py
```

## 6. 类型安全

### 重构前
```python
def calc_one_wind(beta, f_c, h_r, x, y, ...):
    # 没有类型提示
    # IDE无法自动补全
    pass
```

### 重构后
```python
from typing import Optional
from numpy.typing import NDArray

def calculate(self, params: OneWindParameters) -> OneWindResults:
    # 完整的类型提示
    # IDE自动补全
    # 静态类型检查
    pass
```

## 7. 扩展性

### 重构前
```python
# 修改源码才能扩展
def calc_one_wind(...):
    # 硬编码的逻辑
    fourier_result = fourier_solution(...)
    precip = precipitation_grid(...)
    # ...
```

### 重构后
```python
# 插件系统
class CustomPrecipitationSolver(PrecipitationSolver):
    def calculate(self, ...):
        # 自定义实现
        pass

# 注入自定义组件
calculator = OneWindCalculator(config)
calculator._precip_calc = CustomPrecipitationSolver()
```

## 8. 测试友好性

### 重构前
```python
# 难以测试 - 紧耦合
result = calc_one_wind(...)  # 需要所有参数
```

### 重构后
```python
# 容易测试 - 依赖注入

# Mock topography
mock_topo = Topography(...)
calculator.topography = mock_topo

# Mock solver
from unittest.mock import Mock
mock_solver = Mock()
mock_solver.solve.return_value = mock_result
calculator._fourier_solver = mock_solver

# 测试
results = calculator.calculate(params)
```

## 迁移路径

1. **立即使用**: 新模块可以与旧代码并存
2. **逐步迁移**: 逐个功能迁移到新架构
3. **完全切换**: 最终完全使用新架构

```python
# 兼容层示例
from opi.legacy import opi_calc_one_wind  # 旧接口
from opi.calculators import OneWindCalculatorNew  # 新接口

# 两者可以同时使用
old_result = opi_calc_one_wind(...)
new_result = OneWindCalculatorNew(config).calculate(params)
```

## 总结

重构带来的好处：
- ✅ 类型安全 - 减少运行时错误
- ✅ IDE支持 - 更好的开发体验
- ✅ 可测试 - 易于单元测试
- ✅ 可扩展 - 插件化架构
- ✅ 可维护 - 清晰的分层
- ✅ 可配置 - 灵活的配置系统
