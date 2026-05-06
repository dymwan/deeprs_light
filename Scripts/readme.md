# Scripts 使用指南

## acc_ass.py — 遥感精度评估工具

### 功能简介

`acc_ass.py` 是一个面向遥感任务的命令行精度评估工具，支持：

- **矢量输入**：Shapefile (`.shp`), GeoPackage (`.gpkg`), GeoJSON (`.geojson`)
- **栅格输入**：GeoTIFF (`.tif`, `.tiff`)
- **混合模式**：矢量-矢量、栅格-矢量、栅格-栅格
- **四种精度类型**：混淆矩阵 (cm)、目标检测精度 (ap)、几何质量 (go)、位置统计 (pl)
- **自动路径优化**：栅格-栅格模式下 `cm` 和 `go` 走像素级快速计算

### 快速开始

```bash
# 矢量预测 vs 矢量真值，计算混淆矩阵
python Scripts/acc_ass.py \
    --pred pred.shp \
    --gt gt.shp \
    --mode 00 \
    --precision cm \
    --output results.csv

# 栅格预测 vs 矢量真值，计算全部精度
python Scripts/acc_ass.py \
    --pred pred.tif \
    --gt gt.shp \
    --mode 10 \
    --precision all \
    --output results.csv

# 栅格 vs 栅格，计算混淆矩阵 + 几何质量（像素级快速通道）
python Scripts/acc_ass.py \
    --pred pred.tif \
    --gt gt.tif \
    --mode 11 \
    --precision cm,go \
    --band 1 \
    --output results.csv

# 带类别字段的矢量评估
python Scripts/acc_ass.py \
    --pred pred.shp \
    --gt gt.shp \
    --mode 00 \
    --precision cm,pl \
    --field class_id \
    --iou 0.5 \
    --output results.csv
```

### 批量模式 (Batch Mode)

当有多组预测-真值对需要评估时，除了逐对计算精度，还需要所有对的**总体 (overall) 精度**。overall 指标通过累积中间量（TP/FP/FN 计数、面积、距离等）后统一计算，**不是**各 pair 指标的简单平均。

```bash
# 批量模式
python Scripts/acc_ass.py \
    --pairs pairs.csv \
    --precision cm,go \
    --output results.csv
```

**pairs.csv 格式**：

```csv
name,pred,gt,mode
region_1,pred_region1.tif,gt_region1.tif,11
region_2,pred_region2.tif,gt_region2.tif,11
region_3,pred_region3.shp,gt_region3.shp,00
city_a,pred_city.shp,gt_city.shp,00
```

| 列 | 说明 |
|----|------|
| `name` | pair 名称（在输出的 `pair` 列中使用），缺省时自动从 pred 文件名推断 |
| `pred` | 预测结果路径 |
| `gt` | 真值路径 |
| `mode` | 模式字符串 `'xy'`，与单对模式相同 |

#### overall 计算策略

为了保证数学正确性，overall 指标**不通过对各 pair 的结果求平均得到**，而是从原始统计量累加后统一计算：

| 精度 | 累加的中间量 | overall 计算时机 |
|------|------------|----------------|
| `cm` | 逐 pair 累加 per-class TP/FP/FN | 全部 pair 跑完后统一算 P/R/F1/IoU |
| `ap` | 逐 pair 拼接 COCO 预测列表（每条重新分配 image_id） | 全部合并后一次性调 `pycocotools` |
| `go` | 逐 pair 累加 TP/FN + 交集面积 + GT 面积 | 全部跑完后统一算 GTC/GOC/GUC |
| `pl` | 逐 pair 拼接距离数组 | 全部跑完后统计 mean/std/rmse |

这保证了 `cm` 的 macro/micro avg 是基于全局 TP/FP/FN 而非各 pair 的简单数值平均；`ap` 的 mAP 是在完整预测集上统一评估，而非 per-pair 再汇总。

### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--pred` | path | 单对模式 | 预测结果路径 (shp/gpkg/tif) |
| `--gt` | path | 单对模式 | 真值路径 (shp/gpkg/tif) |
| `--mode` | str | 单对模式 | 两位数字字符串 `'xy'`，见下方模式说明 |
| `--pairs` | path | 批量模式 | CSV 文件，列：`name,pred,gt,mode` |
| `--precision` | str | **是** | 精度类型，逗号分隔或 `all`：`cm`, `ap`, `go`, `pl` |
| `--output` | path | **是** | 输出 CSV 路径（仅支持 `.csv` 后缀） |
| `--field` | str | 否 | 类别字段，见下方字段说明 |
| `--band` | int | 否 | 栅格波段号（默认 1），矢量输入忽略 |
| `--iou` | float | 否 | IoU 阈值（默认 0.5） |

### 模式 (mode) 详解

`--mode` 由两个数字组成：`xy`

| 数字 | 输入类型 | 支持的精度 | 说明 |
|------|---------|-----------|------|
| `0` | 矢量 | cm, ap, go, pl | shp/gpkg/geojson，提取要素 bounding box |
| `1` | 二值/多值栅格 | cm, ap, go, pl | tif，像素值 = 类别 ID |
| `2` | 连续分布概率 | 暂不支持 | 未来扩展，目前抛 NotImplementedError |

**已实现模式**：

| 模式 | 预测 | 真值 | 计算策略 |
|------|------|------|---------|
| `00` | 矢量 | 矢量 | 全走矢量级（IoU 匹配 → 现有 evaluator 代码） |
| `10` | 栅格 | 矢量 | 栅格→矢量转换 → 按 `00` 处理 |
| `11` | 栅格 | 栅格 | cm/go 走**像素级快速通道**，ap 需矢量转换 |

### 精度类型详解

#### cm — 混淆矩阵 (Confusion Matrix)

计算指标：per-class + macro/micro avg 的 Precision, Recall, F1, IoU。

- 矢量模式：通过 IoU 匹配 pred-gt box 对，累计 TP/FP/FN
- 栅格-栅格模式：**像素级直接对比**，O(H×W) 复杂度，极快

#### ap — 目标检测精度 (Average Precision)

计算指标：AP, AP50, AP75, AP_s, AP_m, AP_l, AR1, AR10, AR100。

- 基于 COCO 标准，内部调用 `pycocotools`
- 所有模式下都需要矢量格式（栅格自动转换为矢量）

#### go — 几何质量 (Geometric Quality)

计算指标：GTC, GOC, GUC。

| 指标 | 全称 | 含义 |
|------|------|------|
| GTC | Ground Truth Completeness | 地物真值完整性 (= Recall) |
| GOC | Geometric Object Completeness | 几何面积覆盖完整度 |
| GUC | Geometric Usability Completeness | 带可用性阈值的加权 GOC |

- 矢量模式：基于匹配对的 IoU 和面积计算
- 栅格-栅格模式：**像素级直接计算**，O(H×W)，快

#### pl — 点/线位置统计 (PoLis)

计算指标：mean_dist, std_dist, median_dist, max_dist, rmse, buffer_rate。

- 衡量检测目标中心点与真值中心点的定位偏差
- 矢量模式：匹配对的 box 中心点距离
- 栅格模式：连通域中心点距离

### 字段 (field) 详解

```bash
# 二分类：所有要素视为同一类
--field # 不传，默认所有 label=0

# 单一字段：pred 和 gt 用同一个字段名
--field class_id

# 分离字段：逗号分隔，分别对应 pred 和 gt
--field "pred_class,gt_class"
```

### 输出格式 (CSV)

| 列名 | 说明 |
|------|------|
| `precision_type` | 精度类型：`cm`, `ap`, `go`, `pl` |
| `metric` | 指标名称 |
| `class` | 类别标识：`all` 或具体类别名/ID |
| `pair` | pair 标识：单对模式为预测文件名，批量模式为 `name` 列值或 `overall` |
| `value` | 指标值（浮点数保留 6 位小数） |

单对模式示例输出：

```
precision_type,metric,class,pair,value
cm,macro_precision,all,pred_region1,0.785000
cm,macro_recall,all,pred_region1,0.720000
cm,precision_class_ship,ship,pred_region1,0.850000
```

批量模式示例输出（每个 pair 各自的结果 + overall）：

```
precision_type,metric,class,pair,value
cm,macro_precision,all,region_1,0.850000
cm,macro_precision,all,region_2,0.720000
cm,macro_precision,all,region_3,0.810000
cm,macro_precision,all,overall,0.793000
go,GTC,all,region_1,0.750000
go,GTC,all,region_2,0.680000
go,GTC,all,overall,0.715000
```

### 性能说明

栅格-栅格模式 (mode 11) 的像素级计算路径远比矢量化路径快：

| 精度 | mode 11 (像素级) | mode 11 (矢量化) | 速度比 |
|------|-----------------|-----------------|--------|
| `cm` | `np.bincount` 逐像素 | 连通域 → 多边形 → IoU匹配 | **>100x** |
| `go` | 像素统计 | 同上 | **>100x** |
| `pl` | 连通域中心点 | 完整多边形化 | **>10x** |
| `ap` | — (必须矢量化) | 连通域 → COCO → pycocotools | — |

### 依赖

需要额外安装（相比 deeprs_light 核心库）：

```bash
pip install geopandas rasterio fiona shapely
```
