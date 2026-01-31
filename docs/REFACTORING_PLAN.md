# OPI Python 重构计划

## 当前问题分析

### 1. MATLAB风格的问题

| 问题 | 当前实现 | Pythonic方式 |
|:-----|:---------|:-------------|
| 状态管理 | 全局变量/字典传递 | 类实例属性 |
| 参数传递 | 长参数列表 | 配置对象/数据类 |
| 代码组织 | 过程式函数 | 面向对象类 |
| 类型安全 | 无类型提示 | Type hints |
| 扩展性 | 修改源码 | 插件/继承机制 |
| 错误处理 | 简单try-except | 自定义异常类 |

### 2. 具体问题示例

**当前 (MATLAB风格):**
```python
def calc_one_wind(beta, f_c, h_r, x, y, lat, lat0, h_grid, ...):
    # 20+ 个参数
    NM = M * U / h_max
    # ... 大量过程式代码
    return (chi_r2, nu, std_residuals, z_bar, T, ...)  # 元组返回
```

**目标 (Pythonic风格):**
```python
@dataclass
class WindParameters:
    speed: float
    azimuth: float
    temperature: float
    # ...

class OPICalculator:
    def __init__(self, config: OPIConfig):
        self.config = config
        self.results = CalculationResults()
    
    def calculate(self, params: WindParameters) -> CalculationResults:
        # 结构化代码
        pass
```

---

## 重构目标

### 1. 核心架构改进
- [ ] 引入数据类 (`@dataclass`) 管理参数和结果
- [ ] 使用类封装计算逻辑
- [ ] 抽象基类定义接口
- [ ] 插件系统支持扩展

### 2. 代码质量改进
- [ ] 完整的类型提示
- [ ] 自定义异常层次
- [ ] 上下文管理器
- [ ] 迭代器和生成器

### 3. 工程化改进
- [ ] 配置管理系统
- [ ] 日志系统
- [ ] 缓存机制
- [ ] 并行计算支持

---

## 重构路线图

### Phase A: 基础架构 (2-3天)

#### A1. 数据模型层
```
opi/models/
├── __init__.py
├── parameters.py      # 参数数据类
├── results.py         # 结果数据类
├── config.py          # 配置类
└── domain.py          # 网格/域定义
```

#### A2. 核心抽象层
```
opi/core/
├── __init__.py
├── base.py            # 抽象基类
├── calculator.py      # 计算器基类
├── optimizer.py       # 优化器基类
└── interfaces.py      # 接口定义
```

#### A3. 基础设施
```
opi/infrastructure/
├── __init__.py
├── logging.py         # 日志系统
├── config_manager.py  # 配置管理
├── cache.py           # 缓存系统
└── parallel.py        # 并行计算
```

### Phase B: 物理引擎重构 (3-4天)

#### B1. 物理组件类
```python
class FourierSolver:
    """FFT地形求解器"""
    def solve(self, topography: Topography) -> FourierSolution:
        pass

class PrecipitationCalculator:
    """降水计算器"""
    def calculate(self, atmosphere: AtmosphereState) -> PrecipitationGrid:
        pass

class IsotopeCalculator:
    """同位素计算器"""
    def calculate(self, precipitation: PrecipitationGrid) -> IsotopeGrid:
        pass
```

#### B2. 状态管理
```python
@dataclass
class AtmosphereState:
    """大气状态"""
    temperature_profile: TemperatureProfile
    buoyancy_frequency: float
    wind_field: WindField
    
@dataclass  
class WindField:
    """风场"""
    speed: float
    azimuth: float
    direction_vector: Tuple[float, float]
```

### Phase C: 应用层重构 (2-3天)

#### C1. 计算器类
```python
class OneWindCalculator(OPICalculator):
    """单风场计算器"""
    def __init__(self, config: OPIConfig):
        super().__init__(config)
        self.solver = FourierSolver(config.fourier_config)
        self.precip_calc = PrecipitationCalculator()
        self.isotope_calc = IsotopeCalculator()
    
    def calculate(self, params: OneWindParameters) -> OneWindResults:
        # 结构化计算流程
        pass

class TwoWindCalculator(OPICalculator):
    """双风场计算器"""
    def calculate(self, params: TwoWindParameters) -> TwoWindResults:
        # 组合两个风场
        pass
```

#### C2. 优化器类
```python
class CRS3Optimizer(BaseOptimizer):
    """CRS3优化器"""
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.population = Population()
    
    def optimize(self, objective: Callable) -> OptimizationResult:
        # 面向对象优化流程
        pass
```

### Phase D: I/O和数据层 (1-2天)

#### D1. 数据加载器
```python
class DataLoader(ABC):
    """数据加载抽象基类"""
    @abstractmethod
    def load(self, path: Path) -> InputData:
        pass

class MATDataLoader(DataLoader):
    """MAT文件加载器"""
    pass

class ExcelDataLoader(DataLoader):
    """Excel文件加载器"""
    pass
```

#### D2. 结果输出
```python
class ResultExporter(ABC):
    """结果导出抽象基类"""
    @abstractmethod
    def export(self, results: CalculationResults, path: Path):
        pass

class NetCDFExporter(ResultExporter):
    pass

class NumPyExporter(ResultExporter):
    pass
```

### Phase E: 高级功能 (2-3天)

#### E1. 插件系统
```python
class OPIPlugin(ABC):
    """OPI插件基类"""
    @abstractmethod
    def initialize(self, context: OPIContext):
        pass
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        pass

# 示例插件
class CustomFractionationPlugin(OPIPlugin):
    """自定义分馏模型插件"""
    pass
```

#### E2. 工作流系统
```python
class Workflow:
    """计算工作流"""
    def __init__(self):
        self.steps: List[WorkflowStep] = []
    
    def add_step(self, step: WorkflowStep):
        self.steps.append(step)
    
    def execute(self, context: WorkflowContext) -> WorkflowResult:
        for step in self.steps:
            context = step.execute(context)
        return context.result
```

---

## 重构示例

### 重构前
```python
# opi_calc_one_wind.py (当前)
def opi_calc_one_wind(run_file_path=None, solution_vector=None, verbose=True):
    # 加载数据
    input_data = get_input(...)
    
    # 解包参数
    beta = solution_vector or [...]
    
    # 调用计算
    result = calc_one_wind(beta, f_c, h_r, x, y, ...)
    
    # 构造返回
    return {
        'solution_params': ...,
        'results': {...}
    }
```

### 重构后
```python
# opi/calculators/one_wind.py
@dataclass(frozen=True)
class OneWindParameters:
    """单风场参数"""
    wind_speed: float
    azimuth: float
    sea_level_temperature: float
    mountain_height_number: float
    eddy_diffusion: float
    condensation_time: float
    base_d2h: float
    d2h_latitude_gradient: float
    residual_precipitation: float
    
    @property
    def buoyancy_frequency(self) -> float:
        """计算浮力频率"""
        return self.mountain_height_number * self.wind_speed / self.characteristic_height

class OneWindCalculator(OPICalculator):
    """
    单风场OPI计算器
    
    使用示例:
        calculator = OneWindCalculator(config)
        params = OneWindParameters(wind_speed=10.0, ...)
        results = calculator.calculate(params)
    """
    
    def __init__(self, config: OPIConfig):
        super().__init__(config)
        self._fourier_solver = FourierSolver(config.fourier_config)
        self._precipitation_calc = PrecipitationCalculator(config.precip_config)
        self._isotope_calc = IsotopeCalculator(config.isotope_config)
    
    def calculate(self, params: OneWindParameters) -> OneWindResults:
        """
        执行单风场计算
        
        Args:
            params: 风场参数
            
        Returns:
            OneWindResults: 计算结果
            
        Raises:
            CalculationError: 计算失败时抛出
        """
        try:
            with self._profiler.profile("total_calculation"):
                # 1. 准备大气状态
                atmosphere = self._prepare_atmosphere(params)
                
                # 2. FFT求解
                fourier_solution = self._fourier_solver.solve(
                    self.topography, 
                    atmosphere.wind_field
                )
                
                # 3. 计算降水
                precipitation = self._precipitation_calc.calculate(
                    atmosphere,
                    fourier_solution
                )
                
                # 4. 计算同位素
                isotopes = self._isotope_calc.calculate(precipitation, atmosphere)
                
                return OneWindResults(
                    precipitation=precipitation,
                    isotopes=isotopes,
                    atmosphere=atmosphere,
                    metadata=self._create_metadata()
                )
                
        except Exception as e:
            raise CalculationError(f"Calculation failed: {e}") from e
```

---

## 类型提示示例

```python
from typing import Protocol, TypeVar, Generic, Optional, Union
from numpy.typing import NDArray
import numpy as np

T = TypeVar('T', bound='GridData')

class GridData(Protocol):
    """网格数据协议"""
    data: NDArray[np.float64]
    x_coords: NDArray[np.float64]
    y_coords: NDArray[np.float64]
    
    def interpolate(self, x: float, y: float) -> float: ...

class PrecipitationGrid:
    """降水网格"""
    def __init__(self, data: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]):
        self.data = data
        self.x_coords = x
        self.y_coords = y
    
    def interpolate(self, x: float, y: float) -> float:
        # 实现插值
        pass
    
    def to_netcdf(self, path: Path) -> None:
        # 导出到NetCDF
        pass
```

---

## 配置系统示例

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class FourierConfig:
    """FFT求解配置"""
    padding_factor: float = 2.0
    wavenumber_tolerance: float = 1e-10
    handle_singularities: bool = True

@dataclass
class OPIConfig:
    """OPI全局配置"""
    fourier_config: FourierConfig = field(default_factory=FourierConfig)
    parallel: bool = False
    num_workers: int = 1
    cache_enabled: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, path: Path) -> "OPIConfig":
        """从文件加载配置"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

## 重构时间估算

| 阶段 | 时间 | 优先级 |
|:-----|:-----|:-------|
| Phase A: 基础架构 | 2-3天 | P0 |
| Phase B: 物理引擎 | 3-4天 | P0 |
| Phase C: 应用层 | 2-3天 | P0 |
| Phase D: I/O层 | 1-2天 | P1 |
| Phase E: 高级功能 | 2-3天 | P2 |
| **总计** | **10-15天** | |

---

## 重构收益

### 1. 代码质量
- **可读性**: 类和方法名清晰表达意图
- **可维护性**: 模块化设计，易于修改
- **可测试性**: 依赖注入，便于单元测试

### 2. 开发效率
- **IDE支持**: 类型提示提供自动补全
- **错误检测**: 静态类型检查捕获错误
- **文档生成**: 从类型提示生成文档

### 3. 扩展能力
- **插件系统**: 无需修改源码即可扩展
- **配置驱动**: 行为通过配置调整
- **多后端**: 支持不同实现切换

### 4. 性能优化
- **缓存系统**: 自动缓存中间结果
- **并行计算**: 内置并行支持
- **延迟加载**: 按需计算

---

## 建议的重构顺序

1. **先建立数据模型** (parameters.py, results.py)
2. **重构物理引擎** (从FourierSolver开始)
3. **重构计算器** (OneWindCalculator)
4. **添加类型提示**
5. **实现配置系统**
6. **添加高级功能** (插件、工作流)

是否开始进行重构？建议从建立基础架构开始。