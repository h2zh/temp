# 在本地生成照片级模型（3D 高斯泼溅）

原型在服务器上只输出稀疏点云。照片级效果要用 **3D 高斯泼溅（3DGS）** 训练：把场景表示成几十万到上百万个带颜色、半透明的小椭球，反复调整，直到从每张照片的位置看过去都和照片一样。训练需要 GPU，几十分钟能出一个模型。

整个流程三步：**准备"照片 + 相机位姿" → 导出成通用格式 → 用训练工具训练和查看。**

## 第 1 步：装好本仓库

```bash
git clone https://github.com/h2zh/temp && cd temp
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 第 2 步：准备数据（二选一）

### A. 用现成的名胜数据（最快，不用自己算位姿）

MegaDepth-X 数据集里每个名胜都带着真实照片和算好的相机位姿。以佛罗伦萨洗礼堂为例（194 MB，107 张照片）：

```bash
curl -L -o baptistry.tar.gz \
  "https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X/resolve/main/release/Baptistry_(Florence).tar.gz"
mkdir baptistry && tar -xzf baptistry.tar.gz -C baptistry

python scripts/export_for_splatting.py \
  --sparse baptistry/1/sparse --images baptistry/1/images --out splat-baptistry
```

导出后得到 `splat-baptistry/images/` 和 `splat-baptistry/sparse/0/`。这是经典 COLMAP 文本格式、针孔相机，各种训练工具都能直接读。在本仓库的测试机上：107 张照片、约 37 万个初始点，重投影误差 0.76 像素，约 100 MB。

其他名胜换成对应的场景名即可，比如 `Leaning_Tower_(Pisa)`、`Frauenkirche_(Dresden)`。大场景有几 GB；显存不够时加 `--max-images 60 --max-side 1200`，只保留连接最好的 60 张，并把图缩小。

**注意**：这些照片来自 Wikimedia Commons，各有许可（多为 CC BY / CC BY-SA）。自己看没有问题；如果要公开发布用它们训练出的模型，必须遵守署名和相同方式共享等要求。

### B. 用自己拍的照片（效果最好）

绕着一栋建筑走一段，用手机拍 60–150 张照片，相邻两张大约转 5–10°；也可以录 2–3 分钟视频再抽帧。**同一时段拍完、光线一致**，效果比混杂的网络照片好得多。

然后需要算出相机位姿，三种方式任选：
- **Postshot**（Windows + NVIDIA）：直接把照片或视频拖进去，它会自己算位姿并训练，不需要第 3 步。
- **CorbeauSplat**（macOS，免费）：一个界面里先跑 COLMAP、再跑 Brush。
- **COLMAP 图形界面**："Automatic reconstruction" 跑完后，用上面的导出脚本，`--sparse` 指向 `sparse/0`。

**用本项目的流水线**：设置 `Settings(keep_work_dirs=True)`，每个重建作业的位姿会保存在 `data/work/<作业ID>/sparse/0`，照片在 `data/work/<作业ID>/images`。用导出脚本导出即可。

## 第 3 步：训练与查看（按电脑选）

| 你的电脑 | 推荐工具 | 怎么做 |
|---|---|---|
| Mac（Apple 芯片） | [Brush](https://github.com/ArthurBrussee/brush)（免费开源） | 下载对应平台的版本，打开后载入 `splat-baptistry` 文件夹，开始训练，边训练边看效果 |
| Windows + NVIDIA 显卡（≥ 8 GB 显存） | Brush 或 [Postshot](https://www.jawset.com/)（图形界面） | Brush 同上；Postshot 适合方案 B，直接导入照片。注意：从 2026 年 3 月起，Postshot 免费版不能导出 PLY |
| Linux / Windows + NVIDIA，会用命令行 | [gsplat](https://github.com/nerfstudio-project/gsplat)（Apache-2.0，生产方案就用它） | `pip install gsplat`，克隆它的仓库后运行 `python examples/simple_trainer.py default --data_dir splat-baptistry --data_factor 1 --result_dir results`，训练时会给出一个浏览器预览链接 |
| 没有独立显卡 | 租一台云 GPU | 例如按小时租一张 L4（约 $0.8/小时），用 gsplat 训练约半小时，成本 $1 左右 |

训练完得到的 `.ply` 文件，可以拖进 [SuperSplat](https://superspl.at/editor) 网页编辑器查看、裁掉漂浮的杂点，再导出体积更小的格式分享。

## 预期效果

- **网络照片**（方案 A）：拍摄时间、天气、人群都不一样，会出现漂浮的杂点和颜色斑驳。照片级效果只出现在照片多的那几面，洗礼堂主要是东侧。生产方案会给每张照片加"外观编码"并屏蔽路人，来缓解这些问题（见 `docs/ARCHITECTURE.md` 第 3.7 节）。
- **自己一次拍完的照片**（方案 B）：光线一致，通常能得到清晰、可以自由转动的结果。

## 和产品的关系

上线后，这一步会在服务器的 GPU 上自动完成（gsplat 训练，再导出成 SPZ / 3D Tiles 给网页查看器用），用户不需要做任何操作。`scripts/export_for_splatting.py` 就是从现在的流水线（COLMAP 位姿）通向照片级模型的桥。
