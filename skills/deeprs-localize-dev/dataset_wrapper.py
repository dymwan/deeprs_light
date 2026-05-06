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