# ReFlow-VAE-SVC

## 🌟 项目简介

本项目是基于ReFlow-VAE的歌声转换系统，相比原项目，本fork拥有以下增强功能：

### 🔥 主要特性

1. **增强的编码器支持** - 额外支持Whisper-ppg和whisper-ppg-large编码器
2. **友好WebUI界面** - 提供基于Gradio的图形界面，快速上手使用
3. **TensorBoard集成** - 内置训练监控和可视化功能

## 🚀 1. 环境搭建

```bash
git clone https://github.com/RainYooSuki/ReFlow-VAE-SVC.git
```

```bash
pip install -r requirements.txt
```

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📂 2. 预处理数据

### 2.1 音频切片

在进行特征提取之前，建议先对音频进行切片处理，以提高训练效果。

```bash
python slicer.py --input_dir data_raw --output_dir data_sliced
```

### 2.2 数据文件夹结构说明

在进行预处理之前，需要按照以下结构组织您的数据文件夹：

```
ReFlow-VAE-SVC/
├── data/                    # 原始音频文件夹
│   ├── train/
│   │   └── audio/
│             └─── your_speaker_name/
│                       └── audio_file1.wav
│                       └── ...
│   └── val/                     
│       └── audio/
```

#### 进行下一步前请确保：

1. 将原始音频文件放入 `data_raw` 文件夹
2. 运行 `slicer.py` 将长音频切片，结果保存在 `data_sliced` 文件夹
3. 将 `data_sliced` 中的音频文件分别复制到 `data/train/audio` 和 `data/val/audio` 文件夹
4. `pretrain`文件夹中存放好预训练的模型

### 2.3 特征提取

```bash
python draw.py
```

```bash
python preprocess.py -c configs/reflow-vae-wavenet.yaml
```

## 🏋️ 3. 正式训练

```bash
python train.py -c configs/reflow-vae-wavenet.yaml
```

### 3.1. 预训练NSF-HIFIGAN

**新NSF-HIFIGAN**

1. [KOUON PC NSF-HIFIGAN](https://github.com/Kouon-Vocoder-Project/Kouon_Vocoder/releases/download/V2.0.0/kouon_pc_nsf-hifigan_1029_generators.zip)

## ▶️ 4. 非实时推理

### 普通单文件推理模式

需要语义编码器，比如 contentvec：

```bash
python main.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -tid <target_speaker_id> -step <infer_step> -method <method>
```

### VAE 模式

无需语义编码器，特化 sid 到 tid 的变声（或者音高编辑，如果sid == tid）：

```bash
python main.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -sid <source_speaker_id> -tid <target_speaker_id> -step <infer_step> -method <method>
```

## 🌐 5. Web界面操作

本项目提供了一个基于Gradio的Web界面，可以方便地进行音频切片、数据预处理、模型训练和推理操作。

运行以下命令启动Web界面：

```bash
python webui.py
```

启动后，在浏览器中打开 http://localhost:7860 即可访问Web界面。

## 📄 许可证

本项目基于MIT许可证开源，详细信息请查看 [LICENSE](LICENSE) 文件。