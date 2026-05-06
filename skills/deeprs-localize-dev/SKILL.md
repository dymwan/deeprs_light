---
name: deeprs-localize
description: 将第三方库本地化并集成至 deeprs_light 框架，支持论文驱动的 Dataloader 修改与标准化训练脚本构建。
tool_use_rules:
  - 修改代码必须包含 `# MODIFIED BY DEEPRS_LIGHT` 注释。
  - 必须优先使用继承或装饰器进行解耦修改。
  - 默认使用 .pt 格式数据加载（除非显式声明使用 LMDB）。
  - 强制执行 git 分支管理，在 `localize` 分支进行操作。
---

# Skill: DeepRS Localization & Integration

此技能用于引导第三方库（如遥感影像分割、变化检测库）与 `deeprs_light` 框架进行深度融合，确保其符合你的 Registry 注册标准并能在服务器稳定复现。

## 1. 初始化与环境自检
1. **环境确认**：主动询问用户：“当前是否处于可运行环境？”
   - 若用户回答“是”且不在远程服务器，则执行 `python -c "import deeprs_light; print('Success')"` 检查框架可用性。
   - 若在服务器环境，跳过环境检测与本地单元测试。
2. **Git 管理**：检查当前 git 状态，并强制执行：
   ```bash
   git checkout -b localize



## 2. 知识萃取 (Paper-Driven)
1. **文献检索**：搜索项目根目录或 `docs/` 下的 PDF 文件。
2. **Target Key 提取**：若存在论文，重点研读其数据预处理章节，明确：
   - 监督信号的 Key（如 `mask`, `boundary`, `edge`）。
   - 数据类型（int64 为类别，float32 为回归）。
   - 值域范围（如 [0, 1] 或 [0, 255]）。

## 3. 数据层适配 (Data Layer)
1. **存储协议**：除非用户明确要求使用 LMDB，否则默认将数据读取逻辑修改为加载 `.pt` 文件。
2. **Dataloader 修改**：
   - 严格参考 `Reference 1: Dataset Decoupling Wrapper`。
   - 修改 `__getitem__` 返回值，确保 Target Keys 与 `deeprs_light` 的 `Registry` 需求对齐。
3. **离线 Transform**：在当前库的适配层中注册新的数据增强操作。

## 4. 训练与监控集成
1. **构建训练脚本**：创建 `deeprs_light_train.py`，必须参考 `Reference 2: DDP Training Boilerplate`。
2. **多进程安全**：针对 DDP（分布式数据并行）环境，确保 `deeprs_light` 的 Registry 加载和 Monitor 实例不会在多进程中产生竞态冲突。
3. **显存优化**：检查模型 `forward` 过程，排查是否有未释放的中间 Tensor 导致显存泄漏。

## 5. 质量合规与记录
1. **变更文档**：将所有修改点总结写入第三方库根目录的 `deeprs_light_changes.md`，参考 `Reference 3: Changes Tracker Template`。
2. **注释规范**：所有新增或修改的代码行上方或行尾，必须添加 `# MODIFIED BY DEEPRS_LIGHT`。

---

## References (代码参考模板)

当需要修改代码时，严格按照以下模板的结构和规范生成代码。

### Reference 1: Dataset Decoupling Wrapper
**场景**：当需要修改第三方库的数据加载逻辑时，不要直接改源码，而是新建一个包装类。

```python
import torch
# 假设从第三方库导入原始 Dataset
from third_party_repo.datasets import OriginalDataset 
from deeprs_light.registry import DATASETS

@DATASETS.register_module()
class DeepRSWrappedDataset(OriginalDataset):
    """
    通过继承解耦对第三方库 Dataset 的修改。
    """
    def __init__(self, *args, **kwargs):
        # MODIFIED BY DEEPRS_LIGHT: 默认强制使用 .pt 格式
        self.use_pt = kwargs.pop('use_pt', True) 
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # 获取原始数据字典
        data = super().__getitem__(idx)
        
        # MODIFIED BY DEEPRS_LIGHT: 重新映射 Target Keys 并规范数据类型
        # 例：将原始的 'label' 映射为 deeprs_light 规定的 'mask'
        formatted_data = {
            'image': data['img'],
            # 强制类别标签为 int64，回归任务为 float32
            'mask': data['label'].to(torch.int64) 
        }
        return formatted_data
  ```