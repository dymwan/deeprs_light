# DeepRS-Light 本地化修改记录

| 修改文件路径 | 修改方式 (继承/装饰器/直接修改) | 涉及逻辑 | 具体说明 |
| :--- | :--- | :--- | :--- |
| `src/dataset/wrapper.py` | 继承 (`DeepRSWrappedDataset`) | Dataloader 返回值 | 将 `label` 变更为 `mask`，数据格式转为 `int64`，默认加载 `.pt` 文件。 |
| `deeprs_light_train.py` | 新增文件 | DDP 训练流 | 按照框架标准重写了训练脚本，仅在主进程启动 Monitor。 |