# ReFlow-VAE-SVC

## 1.环境搭建

```bash
git clone https://github.com/yxlllc/ReFlow-VAE-SVC
```

```bash
cd ReFlow-VAE-SVC
```

```bash
python -m venv venv
```

```bash
./venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## 2.预处理数据

```bash
python preprocess.py -c configs/reflow-vae-wavenet.yaml
```

## 3.正式训练

```bash
python train.py -c configs/reflow-vae-wavenet.yaml
```

### 3.1. 可下载底模与新NSF-HIFIGAN

**(1).底模**

[wavenet的Beta版底模](https://huggingface.co/OOPPEENN/pretrained_model)

[lynxnet的Beta版底模](https://huggingface.co/tepetst3033/Reflow_VAE_SVC_retrained_model_with_lynxnet)

**(2).新NSF-HIFIGAN**

[Kouon NSF-HIFIGAN](https://github.com/Kouon-Vocoder-Project/Kouon_Vocoder/releases/download/V2.0.0/kouon_pc_nsf-hifigan_1029_generators.zip)

## 4.非实时推理：

```bash
# 普通单文件推理模式, 需要语义编码器, 比如 contentvec
python main.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -tid <target_speaker_id> -step <infer_step> -method <method>
```

```bash
# 普通批量推理模式, 需要语义编码器, 比如 contentvec
python infers.py -m <model_ckpt.pt> -method <methods>
```

```bash
# VAE 模式, 无需语义编码器, 特化 sid 到 tid 的变声（或者音高编辑，如果sid == tid）
python main.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -sid <source_speaker_id> -tid <target_speaker_id> -step <infer_step> -method <method>
```
