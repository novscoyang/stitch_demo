# DJI 风机叶片全景拼接

这个项目用于把一组 DJI 航拍照片拼接成风机叶片全景图。脚本会读取照片中的 GPS EXIF 信息和 DJI XMP 云台姿态数据，使用相机 yaw 和 pitch 将 GPS 位移投影到图像平面，再结合 ORB 特征匹配进行配准和合成。

## 项目结构

```text
.
├── input/                         # 原始 DJI 照片
├── outputs/                       # 生成的拼接结果
├── stitch_blade_panorama.py       # 主拼接脚本
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

## 环境要求

- Python 3.10 或更高版本
- 带 GPS EXIF 信息的 DJI 照片
- 照片中最好包含 DJI XMP 字段，例如 `GimbalYawDegree` 和 `GimbalPitchDegree`

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 快速开始

推荐使用：
 python stitch_blade_panorama.py --input input --output outputs/stitched_input_direct_camera_pitch_x14_refined.png --full-frame --gps-x-scale 3.0

使用默认相机姿态投影生成完整画面直贴图：

```powershell
python stitch_blade_panorama.py --input input --output outputs/stitched_input_direct_camera_pitch.png --full-frame
```

只合成识别到的叶片区域：

```powershell
python stitch_blade_panorama.py --input input --output outputs/blade_panorama.png
```

生成低分辨率预览图：

```powershell
python stitch_blade_panorama.py --input input --output outputs/preview.png --scale 800 --low-res-output
```

## 处理流程

1. 从 `input/` 读取原始图片。
2. 解析 GPS 经纬度、高度、35mm 等效焦距，以及 DJI 云台 yaw/pitch。
3. 将 GPS 坐标转换成局部 ENU 坐标。
4. 使用相机 yaw 和 pitch 将 ENU 位移投影到图像方向。
5. 使用 ORB 特征匹配修正相邻图片的位移和旋转。
6. 根据参数选择合成叶片 mask 区域，或直接贴入完整原始帧。

默认的 `--gps-projection camera` 会使用 DJI 相机 yaw 和 pitch。`--gps-projection enu` 主要用于调试，它会直接使用原始 ENU 东北方向位移。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | `input` | 输入图片目录。 |
| `--output` | `stitched_blade_panorama.png` | 输出图片路径。父目录不存在时会自动创建。 |
| `--scale` | `1500` | 配准时使用的缩放宽度。设为 `0` 表示用原始分辨率配准。 |
| `--full-frame` | 关闭 | 贴入完整原始帧，而不是只合成叶片 mask 区域。 |
| `--low-res-output` | 关闭 | 输出缩放后的配准图，而不是原始分辨率合成图。 |
| `--blend-mode` | `direct` | 原始分辨率叶片合成模式，可选 `direct` 或 `average`。 |
| `--gsd-mode` | `exif` | GPS 到像素比例来源，可选 `exif` 或 `visual`。 |
| `--gps-projection` | `camera` | GPS 投影模式。`camera` 使用 yaw/pitch，`enu` 使用原始 ENU 轴。 |
| `--gps-x-scale` | `1.0` | 额外放大或缩小投影后的 x 方向 GPS 位移。dx 方向太密时可调大。 |
| `--gps-y-sign` | `invert` | GPS 投影到图像 y 方向时的符号。 |
| `--no-visual-refine` | 关闭 | 禁用视觉平移微调，只使用 GPS 投影位移。 |
| `--max-visual-shift` | `40` | 每对图片允许视觉算法修正的最大像素量，单位是缩放后的配准像素。 |
| `--visual-weight` | `0.35` | 视觉修正权重。越大越相信视觉匹配，越小越相信 GPS。 |
| `--mask-percentile` | `12` | 叶片低饱和度 mask 的百分位阈值。 |
| `--min-mask-ratio` | `0.02` | 小于该 mask 面积比例的帧会被跳过。 |
| `--save-mask-preview` | 关闭 | 在输出图片旁边保存叶片 mask 预览图。 |

## 输出说明

- 建议把生成结果统一放在 `outputs/`。
- 当前示例输出为 `outputs/stitched_input_direct_camera_pitch.png`。
- 如果使用 `--save-mask-preview`，会额外生成同名 `_masks.jpg` 预览图。

## 调试建议

- 如果拼接方向明显不对，先比较 `--gps-projection camera` 和 `--gps-projection enu`。
- 如果 dx 方向太密，优先尝试 `--gps-x-scale 1.3` 到 `--gps-x-scale 1.5`。
- 默认会在 GPS 初值基础上做保守视觉微调，用于减少局部错位；如果视觉匹配导致跳变，可加 `--no-visual-refine`。
- 如果局部错位仍明显，可小幅增大 `--max-visual-shift` 或 `--visual-weight`，例如 `--max-visual-shift 60 --visual-weight 0.45`。
- 如果 GPS 到像素比例整体偏差较大，可以尝试 `--gsd-mode visual`，但它可能会过度拉开 x 方向，需要结合预览判断。
- 如果 y 方向整体翻转，可以尝试 `--gps-y-sign same`。
- 如果只想检查位姿和全局排列，用 `--full-frame` 更直观。
- 如果只关心叶片输出，不要加 `--full-frame`。

## 注意事项

- `input/` 中应放同一段连续拍摄的 DJI 原图。
- 脚本依赖照片中的 GPS 和 DJI 云台元数据；缺失元数据会影响自动投影效果。
- 生成图、调试图、缓存和日志已在 `.gitignore` 中忽略。
- 原始照片通常体积较大，是否纳入版本管理应根据实际项目需要决定。


