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

### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--pred` | path | 是 | 预测结果路径 (shp/gpkg/tif) |
| `--gt` | path | 是 | 真值路径 (shp/gpkg/tif) |
| `--mode` | str | 是 | 两位数字字符串 `'xy'`，见下方模式说明 |
| `--precision` | str | 是 | 精度类型，逗号分隔或 `all`：`cm`, `ap`, `go`, `pl` |
| `--output` | path | 是 | 输出 CSV 路径（仅支持 `.csv` 后缀） |
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
| `value` | 指标值（浮点数保留 6 位小数） |

示例输出：

```
precision_type,metric,class,value
cm,macro_precision,all,0.785000
cm,macro_recall,all,0.720000
cm,precision_class_ship,ship,0.850000
ap,AP,all,0.450000
ap,AP50,all,0.720000
go,GTC,all,0.750000
go,GOC,all,0.650000
pl,PoLis_mean_dist,all,2.340000
pl,PoLis_rmse,all,2.910000
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
