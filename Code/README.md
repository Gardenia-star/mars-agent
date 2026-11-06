# MARS-Agent 火眼检测系统

基于 YOLOv10 和 ResNet18 的铝电解槽火眼过热度智能检测系统

## 📁 项目结构

```
Mars_Agent/
├── qt_interface/              # PyQt5 桌面界面
│   ├── ui_fireeye.py         # 主界面程序
│   ├── algo_fireeye.py       # 核心算法模块
│   ├── llm_service.py        # 大模型分析服务
│   └── history_record.json   # 历史记录配置
├── web_interface/             # Streamlit Web界面
│   ├── streamlit_app.py      # Web应用主程序
│   ├── algo_web_bridge.py    # 算法桥接层
│   └── llm_web.py           # Web端大模型服务
├── weights/                   # 模型权重文件
│   ├── best.pt              # YOLOv10 目标检测权重
│   └── resnet18_fireeye_3cls.pth  # ResNet18 分类权重
├── config/                    # 配置文件目录
├── assets/                    # 静态资源目录
├── requirements.txt          # Python依赖清单
└── README.md                 # 本文档
```

## 🚀 快速开始

### 1. 环境准备

**系统要求：**
- Python 3.8 或以上版本
- Windows 10/11 或 Linux
- （推荐）NVIDIA GPU + CUDA 11.8+

**安装依赖：**

```bash
# 进入项目目录
cd Mars_Agent

# 安装Python依赖包
pip install -r requirements.txt
```

### 2. 模型权重配置

项目已包含训练好的权重文件：
- `weights/best.pt` - YOLOv10火眼目标检测模型
- `weights/resnet18_fireeye_3cls.pth` - ResNet18三分类模型（过冷/正常/过热）

**注意：** 请确保权重文件路径在代码中正确配置

### 3. 大模型服务（可选）

如需使用AI智能分析功能，需要安装 Ollama 并下载 Llava 模型：

```bash
# 安装 Ollama (https://ollama.ai)
# Windows: 下载安装包
# Linux: curl https://ollama.ai/install.sh | sh

# 拉取 Llava 模型
ollama pull llava

# 启动 Ollama 服务
ollama serve
```

## 💻 使用 PyQt5 桌面界面

### 启动方式

```bash
cd qt_interface
python ui_fireeye.py
```

### 功能说明

#### 1. 主界面布局
- **左侧区域**：实时视频显示和ROI检测区域
- **右侧区域**：统计图表、历史记录、参数显示
- **底部区域**：控制按钮和状态栏

#### 2. 核心功能

**视频源选择**
- 本地视频文件：点击"打开视频"按钮选择MP4/AVI格式文件
- 实时摄像头：选择摄像头编号（0, 1, 2...）
- RTSP网络流：输入RTSP地址（支持海康/大华等）

**检测模式**
- **目标检测**：使用YOLO模型定位火眼区域
- **状态分类**：使用ResNet对检测到的区域进行三分类
  - 🔵 C (Cold) - 偏低/过冷
  - 🟢 N (Normal) - 正常
  - 🔴 H (Hot) - 偏高/过热

**实时分析**
- 平均灰度 μ：反映整体亮度
- 对比度 σ：反映纹理变化
- 饱和度 S：反映颜色强度
- 结壳速率 V_crust：动态特征
- 角点速率 V_node：边缘变化

**AI大模型分析**
- 点击"AI分析"按钮调用 Llava 多模态大模型
- 获取专家级诊断报告和操作建议
- 支持中文工艺术语理解

**历史记录**
- 自动记录每次检测结果
- 可查看历史快照和分析报告
- 支持导出为JSON格式

#### 3. 操作流程

1. **启动程序**：运行 `python ui_fireeye.py`
2. **选择视频源**：点击"打开视频"或选择摄像头
3. **开始检测**：点击"开始"按钮
4. **查看结果**：观察实时检测框、分类结果和统计图表
5. **AI分析**（可选）：点击"AI分析"获取详细报告
6. **暂停/停止**：使用底部控制按钮
7. **查看历史**：右侧历史记录列表双击查看详情

#### 4. 快捷键

- `Space` - 暂停/继续
- `Esc` - 停止检测
- `Ctrl+O` - 打开视频文件
- `Ctrl+S` - 保存当前帧

## 🌐 使用 Streamlit Web 界面

### 启动方式

```bash
cd web_interface
streamlit run streamlit_app.py
```

浏览器会自动打开 `http://localhost:8501`

### 功能说明

#### 1. 页面布局
- **顶部**：标题和时间显示
- **左侧边栏**：上传文件和参数配置
- **主区域**：图像显示和分析结果

#### 2. 使用流程

1. **启动服务**：运行 streamlit 命令
2. **上传图片**：点击侧边栏"上传图片"按钮
3. **选择模型**：选择 ResNet 或其他模型
4. **查看结果**：
   - 左侧显示原始图像
   - 右侧显示检测结果和特征参数
5. **AI分析**：点击"生成AI报告"按钮获取智能分析

#### 3. Web界面特点

- ✅ 无需安装客户端，浏览器直接访问
- ✅ 支持拖拽上传图片
- ✅ 实时显示分析结果
- ✅ 响应式布局，支持移动端
- ✅ 多用户同时访问（session隔离）

#### 4. 网络部署

**局域网访问：**
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
然后在局域网内通过 `http://<服务器IP>:8501` 访问

**生产环境部署：**
建议使用 Nginx 反向代理 + SSL 证书

## 🔧 配置说明

### 修改权重路径

编辑 `qt_interface/algo_fireeye.py` 或 `web_interface/algo_fireeye.py`：

```python
# 第 18-19 行
YOLO_WEIGHT = r"../weights/best.pt"
RESNET_WEIGHT = r"../weights/resnet18_fireeye_3cls.pth"
```

### 调整检测参数

```python
# algo_fireeye.py 中的参数
CONF_EYE = 0.18      # YOLO置信度阈值
CONF_FIRE = 0.15     # 火焰检测阈值
MIN_SIDE = 20        # 最小目标尺寸
```

### 配置大模型

```python
# llm_service.py 或 llm_web.py
model_name = "llava"  # 可更换为其他Ollama模型
```

## 📊 算法说明

### 1. 两阶段检测流程

**Stage 1: YOLO目标检测**
- 输入：视频帧
- 输出：火眼区域边界框（bbox）
- 类别：eye（火眼）、fire（火焰）

**Stage 2: ResNet分类**
- 输入：裁剪的ROI区域
- 输出：三分类结果（C/N/H）
- 特征：灰度、对比度、饱和度、动态速率

### 2. 特征计算

```
平均灰度 μ = mean(gray_image)
对比度 σ = std(gray_image)
饱和度 S = mean(hsv[:,:,1]) / 255
结壳速率 V_crust = 边缘变化速率
角点速率 V_node = 角点位移速率
```

### 3. 分类规则

- **C (Cold/偏低)**：灰度低、饱和度低、冷色调
- **N (Normal/正常)**：灰度中等、特征均衡
- **H (Hot/过热)**：灰度高、饱和度高、暖色调

## ⚠️ 常见问题

### 1. 导入错误

**问题：** `ModuleNotFoundError: No module named 'xxx'`

**解决：**
```bash
pip install -r requirements.txt
```

### 2. 权重文件加载失败

**问题：** `FileNotFoundError: weights/best.pt not found`

**解决：**
- 检查 `weights/` 目录是否包含权重文件
- 修改代码中的路径为绝对路径
- 确保权重文件没有损坏

### 3. CUDA 内存不足

**问题：** `RuntimeError: CUDA out of memory`

**解决：**
- 减小批处理大小
- 降低输入图像分辨率
- 切换到CPU模式（性能会降低）：
  ```python
  device = torch.device('cpu')
  ```

### 4. Ollama 连接失败

**问题：** `Connection refused` 或 `Model not found`

**解决：**
- 确保 Ollama 服务正在运行：`ollama serve`
- 检查模型是否已下载：`ollama list`
- 如未下载：`ollama pull llava`

### 5. Qt界面显示异常

**问题：** 界面控件错位或无法显示

**解决：**
```bash
# Windows
pip uninstall PyQt5 PyQt5-sip
pip install PyQt5==5.15.9 PyQt5-sip==12.11.0

# Linux需要安装额外依赖
sudo apt-get install python3-pyqt5
```

### 6. Streamlit 无法启动

**问题：** `Address already in use`

**解决：**
```bash
# 更换端口
streamlit run streamlit_app.py --server.port 8502
```

## 📝 更新日志

### v1.0.0 (2025-11-26)
- ✅ 完成 PyQt5 桌面界面
- ✅ 集成 YOLOv10 + ResNet18 双模型
- ✅ 实现实时视频检测
- ✅ 添加 Llava 大模型分析
- ✅ 完成 Streamlit Web 界面
- ✅ 历史记录功能
- ✅ 多维度特征提取

## 📧 技术支持

如有问题，请联系项目维护者。

## 📄 许可证

本项目仅供学习和研究使用。

---

**祝使用愉快！🎉**
