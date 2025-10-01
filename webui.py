# coding: utf-8

import os
import sys
import shutil
import argparse
import tempfile
import torch
import librosa
import soundfile as sf
import numpy as np
import gradio as gr
from slicer import Slicer, cut, chunks2audio
from preprocess import preprocess
from reflow.extractors import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import Vocoder, load_model_vocoder
from logger import utils
import subprocess
import yaml
import threading
import signal
import psutil
import time

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(__file__))


def slice_audio(threshold, min_length, min_interval, hop_size, max_sil_kept):
    """
    音频切片功能
    """
    try:
        # 创建输出目录
        output_dir = "data_sliced"
        os.makedirs(output_dir, exist_ok=True)

        # 支持的音频格式
        audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']

        # 获取所有音频文件
        input_dir = "data_raw"
        audio_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(file)

        if not audio_files:
            return "输入文件夹中没有找到音频文件", None

        total_segments = 0

        # 处理每个音频文件
        for audio_file in audio_files:
            file_path = os.path.join(input_dir, audio_file)
            file_name = os.path.splitext(audio_file)[0]

            # 加载音频
            audio, sample_rate = librosa.load(file_path, sr=None)

            # 创建Slicer对象
            slicer = Slicer(
                sr=sample_rate,
                threshold=threshold,
                min_length=min_length,
                min_interval=min_interval,
                hop_size=hop_size,
                max_sil_kept=max_sil_kept
            )

            # 执行切片
            chunks = slicer.slice(audio)

            # 保存切片结果
            file_segments = 0
            for i, (chunk_key, chunk_data) in enumerate(chunks.items()):
                if not chunk_data["slice"]:  # 只保存非静音片段
                    split_time = chunk_data["split_time"].split(",")
                    start = int(split_time[0])
                    end = int(split_time[1])

                    # 提取音频片段
                    if len(audio.shape) > 1:
                        segment = audio[:, start:end]
                    else:
                        segment = audio[start:end]

                    # 保存音频片段
                    output_path = os.path.join(output_dir, f"{file_name}_seg{i:04d}.wav")
                    sf.write(output_path, segment, sample_rate)
                    file_segments += 1
                    total_segments += 1

            print(f"已处理 {audio_file}，生成 {file_segments} 个片段")

        message = f"切片完成，共处理 {len(audio_files)} 个音频文件，生成 {total_segments} 个片段"

        return message, output_dir
    except Exception as e:
        return f"处理过程中出现错误: {str(e)}", None


# 全局变量用于存储进程引用
preprocess_process = None
train_process = None


# 停止进程的函数
def stop_process(process):
    if process and process.poll() is None:  # 检查进程是否仍在运行
        try:
            # 获取进程树并终止所有相关进程
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            return True
        except psutil.NoSuchProcess:
            return False
    return False


def run_draw():
    """
    运行draw.py脚本
    """
    try:
        # 使用sys.executable确保使用当前Python环境
        cmd = [sys.executable, 'draw.py']

        # 执行draw.py脚本
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode == 0:
            return f"draw.py执行成功！\n\n输出:\n{result.stdout}"
        else:
            return f"draw.py执行过程中出现错误:\n{result.stderr}"
    except Exception as e:
        return f"执行draw.py时出现错误: {str(e)}"


def run_preprocess(config_path):
    """
    数据预处理功能
    """
    global preprocess_process
    try:
        # 构建命令行参数
        cmd = [
            sys.executable, 'preprocess.py',
            '--config', config_path
        ]

        # 执行预处理脚本
        preprocess_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                              cwd=os.getcwd())
        stdout, stderr = preprocess_process.communicate()

        if preprocess_process.returncode == 0:
            return f"预处理成功完成！\n\n输出:\n{stdout}"
        else:
            return f"预处理过程中出现错误:\n{stderr}"
    except Exception as e:
        return f"执行预处理时出现错误: {str(e)}"


def stop_preprocess():
    """
    停止数据预处理
    """
    global preprocess_process
    if stop_process(preprocess_process):
        return "预处理已停止"
    else:
        return "没有正在运行的预处理任务"


def run_training(config_path):
    """
    模型训练功能
    """
    global train_process
    try:
        # 构建命令行参数
        cmd = [
            sys.executable, 'train.py',
            '--config', config_path
        ]

        # 执行训练脚本
        train_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                         cwd=os.getcwd())
        stdout, stderr = train_process.communicate()

        if train_process.returncode == 0:
            return f"训练启动成功！\n\n输出:\n{stdout}"
        else:
            return f"训练启动过程中出现错误:\n{stderr}"
    except Exception as e:
        return f"执行训练时出现错误: {str(e)}"


def stop_training():
    """
    停止模型训练
    """
    global train_process
    if stop_process(train_process):
        return "训练已停止"
    else:
        return "没有正在运行的训练任务"


def run_inference(model_ckpt, input_audio, output_dir, key, speaker_id, infer_step, config_path):
    """
    音频推理功能
    """
    try:
        # 检查输入音频是否为空
        if input_audio is None:
            return "请先上传音频文件", None

        # 保存上传的音频，使用更精确的方式处理文件大小
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            # input_audio是一个元组，第一个元素是采样率，第二个是音频数据
            # 我们需要将音频数据写入临时文件
            import soundfile as sf
            # 确保数据类型正确，避免编码问题
            audio_data = input_audio[1].astype(np.float32)
            sf.write(temp_input.name, audio_data, input_audio[0])
            temp_input_path = temp_input.name

        # 确保outputs文件夹存在
        if not output_dir or output_dir == "outputs":
            output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名：在原始文件名后添加"_output"后缀
        input_filename = os.path.basename(temp_input_path)
        filename_without_ext, ext = os.path.splitext(input_filename)
        output_filename = f"{filename_without_ext}_output{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # 构建命令行参数
        cmd = [
            sys.executable, 'main.py',
            '--model_ckpt', model_ckpt,
            '--input', temp_input_path,
            '--output', output_path,
            '--key', str(key),
            '--target_spk_id', str(speaker_id),
            '--infer_step', str(infer_step)
        ]

        if config_path:
            # 加载配置以获取默认参数
            try:
                args = utils.load_config(config_path)
                cmd.extend([
                    '--pitch_extractor', args.infer.pitch_extractor or 'rmvpe',
                    '--f0_min', str(args.data.f0_min or 50),
                    '--f0_max', str(args.data.f0_max or 1100)
                ])
            except:
                cmd.extend([
                    '--pitch_extractor', 'rmvpe',
                    '--f0_min', '50',
                    '--f0_max', '1100'
                ])

        # 执行推理脚本并实时输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   cwd=os.getcwd(), encoding='utf-8')
        output_lines = []

        # 实时读取输出
        for line in process.stdout:
            output_lines.append(line)
            # 这里可以考虑使用回调函数更新界面，但Gradio不直接支持
            # 目前我们收集所有输出并在最后返回

        # 等待进程结束
        process.wait()

        # 清理临时文件
        os.unlink(temp_input_path)

        output_text = ''.join(output_lines)

        if process.returncode == 0:
            if os.path.exists(output_path):
                return f"推理成功完成！\n\n输出日志:\n{output_text}", output_path
            else:
                return f"推理完成但未找到输出文件:\n\n输出日志:\n{output_text}", None
        else:
            return f"推理过程中出现错误:\n\n输出日志:\n{output_text}", None
    except Exception as e:
        # 添加更详细的错误信息
        import traceback
        error_details = traceback.format_exc()
        return f"执行推理时出现错误: {str(e)}\n详细信息:\n{error_details}", None


def get_config_files():
    """
    获取配置文件列表
    """
    config_dir = "configs"
    if os.path.exists(config_dir):
        return [os.path.join(config_dir, f) for f in os.listdir(config_dir) if
                f.endswith('.yaml') or f.endswith('.yml')]
    return []


def load_config_data(config_path):
    """
    加载配置文件数据
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return config_data
    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        return None


def save_config_data(config_path, config_data):
    """
    保存配置文件数据
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"保存配置文件时出错: {str(e)}")
        return False


def get_model_files():
    """
    获取模型文件列表
    """
    model_dirs = ["exp"]  # 只从exp文件夹读取模型
    model_files = []
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pt') or file.endswith('.pth'):
                        model_files.append(os.path.join(root, file))
    return model_files


def get_exp_config_files():
    """
    获取exp文件夹中的配置文件列表
    """
    exp_dir = "exp"
    config_files = []
    if os.path.exists(exp_dir):
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    config_files.append(os.path.join(root, file))
    return config_files


with gr.Blocks(title="ReFlow VAE SVC WebUI") as app:
    gr.Markdown("# ReFlow VAE SVC WebUI")
    gr.Markdown("一个基于Gradio的图形界面，用于音频切片、预处理、训练和推理")

    with gr.Tab("音频切片"):
        with gr.Row():
            with gr.Column():
                slice_threshold = gr.Number(label="静音阈值 (dB)", value=-40, info="越低则切得越细")
                slice_min_length = gr.Number(label="最小音频长度 (ms)", value=5000)
                slice_min_interval = gr.Number(label="最小静音间隔 (ms)", value=300)
                slice_hop_size = gr.Number(label="Hop size (ms)", value=20)
                slice_max_sil_kept = gr.Number(label="最大静音保留 (ms)", value=5000)
                slice_button = gr.Button("开始切片")
            with gr.Column():
                slice_output = gr.Textbox(label="切片结果", lines=5, max_lines=10)
                slice_output_dir = gr.Textbox(label="输出目录", lines=3, max_lines=5)

    slice_button.click(
        slice_audio,
        inputs=[slice_threshold, slice_min_length, slice_min_interval, slice_hop_size, slice_max_sil_kept],
        outputs=[slice_output, slice_output_dir]
    )

    with gr.Tab("数据预处理"):
        with gr.Row():
            with gr.Column():
                draw_button = gr.Button("运行Draw (抽取验证集)")

                preprocess_config = gr.Dropdown(
                    choices=get_config_files(),
                    label="选择配置文件",
                    value=get_config_files()[0] if get_config_files() else None
                )
                refresh_preprocess_config = gr.Button("刷新配置文件列表")
                preprocess_button = gr.Button("开始预处理")
                stop_preprocess_button = gr.Button("停止预处理")
            with gr.Column():
                preprocess_output = gr.Textbox(label="预处理输出", lines=10, max_lines=20)

    draw_button.click(
        run_draw,
        outputs=[preprocess_output]
    )

    refresh_preprocess_config.click(
        lambda: gr.Dropdown(choices=get_config_files()),
        outputs=[preprocess_config]
    )

    preprocess_button.click(
        run_preprocess,
        inputs=[preprocess_config],
        outputs=[preprocess_output]
    )

    stop_preprocess_button.click(
        stop_preprocess,
        outputs=[preprocess_output]
    )

    with gr.Tab("模型训练"):
        with gr.Row():
            with gr.Column():
                train_config = gr.Dropdown(
                    choices=get_config_files(),
                    label="选择配置文件",
                    value=get_config_files()[0] if get_config_files() else None
                )
                refresh_train_config = gr.Button("刷新配置文件列表")

                # 添加模型参数显示和修改
                gr.Markdown("### 模型参数")
                train_n_layers = gr.Number(label="n_layers", value=32)
                train_n_chans = gr.Number(label="n_chans", value=1280)
                train_n_hidden = gr.Number(label="n_hidden", value=512)

                gr.Markdown("### 训练参数")
                train_num_workers = gr.Number(label="num_workers", value=0)
                train_batch_size = gr.Number(label="batch_size", value=48)
                train_epochs = gr.Number(label="epochs", value=100000)
                train_lr = gr.Number(label="lr", value=0.0001)
                train_decay_step = gr.Number(label="decay_step", value=100000)
                train_gamma = gr.Number(label="gamma", value=0.5)
                train_weight_decay = gr.Number(label="weight_decay", value=0)

                load_config_button = gr.Button("加载配置")
                save_config_button = gr.Button("保存配置")
                train_button = gr.Button("开始训练")
                stop_train_button = gr.Button("停止训练")
            with gr.Column():
                train_output = gr.Textbox(label="训练输出", lines=10, max_lines=20)


    def load_train_config(config_path):
        """
        加载训练配置
        """
        config_data = load_config_data(config_path)
        if config_data is None:
            return [gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update()]

        # 获取模型参数
        model_params = config_data.get('model', {})
        n_layers = model_params.get('n_layers', 32)
        n_chans = model_params.get('n_chans', 1280)
        n_hidden = model_params.get('n_hidden', 512)

        # 获取训练参数
        train_params = config_data.get('train', {})
        num_workers = train_params.get('num_workers', 0)
        batch_size = train_params.get('batch_size', 48)
        epochs = train_params.get('epochs', 100000)
        lr = train_params.get('lr', 0.0001)
        decay_step = train_params.get('decay_step', 100000)
        gamma = train_params.get('gamma', 0.5)
        weight_decay = train_params.get('weight_decay', 0)

        return [
            gr.update(value=n_layers),
            gr.update(value=n_chans),
            gr.update(value=n_hidden),
            gr.update(value=num_workers),
            gr.update(value=batch_size),
            gr.update(value=epochs),
            gr.update(value=lr),
            gr.update(value=decay_step),
            gr.update(value=gamma),
            gr.update(value=weight_decay)
        ]


    def save_train_config(config_path, n_layers, n_chans, n_hidden, num_workers, batch_size, epochs, lr, decay_step,
                          gamma, weight_decay):
        """
        保存训练配置
        """
        config_data = load_config_data(config_path)
        if config_data is None:
            return "加载配置文件失败"

        # 更新模型参数
        if 'model' not in config_data:
            config_data['model'] = {}
        config_data['model']['n_layers'] = int(n_layers)
        config_data['model']['n_chans'] = int(n_chans)
        config_data['model']['n_hidden'] = int(n_hidden)

        # 更新训练参数
        if 'train' not in config_data:
            config_data['train'] = {}
        config_data['train']['num_workers'] = int(num_workers)
        config_data['train']['batch_size'] = int(batch_size)
        config_data['train']['epochs'] = int(epochs)
        config_data['train']['lr'] = float(lr)
        config_data['train']['decay_step'] = int(decay_step)
        config_data['train']['gamma'] = float(gamma)
        config_data['train']['weight_decay'] = float(weight_decay)

        # 保存配置文件
        if save_config_data(config_path, config_data):
            return f"配置已保存到 {config_path}"
        else:
            return "保存配置文件失败"


    refresh_train_config.click(
        lambda: gr.Dropdown(choices=get_config_files()),
        outputs=[train_config]
    )

    load_config_button.click(
        load_train_config,
        inputs=[train_config],
        outputs=[train_n_layers, train_n_chans, train_n_hidden, train_num_workers, train_batch_size, train_epochs,
                 train_lr, train_decay_step, train_gamma, train_weight_decay]
    )

    save_config_button.click(
        save_train_config,
        inputs=[train_config, train_n_layers, train_n_chans, train_n_hidden, train_num_workers, train_batch_size,
                train_epochs, train_lr, train_decay_step, train_gamma, train_weight_decay],
        outputs=[train_output]
    )

    train_button.click(
        run_training,
        inputs=[train_config],
        outputs=[train_output]
    )

    stop_train_button.click(
        stop_training,
        outputs=[train_output]
    )

    with gr.Tab("音频推理"):
        with gr.Row():
            with gr.Column():
                infer_model = gr.Dropdown(
                    choices=get_model_files(),
                    label="选择模型文件 (仅exp文件夹)",
                    value=get_model_files()[0] if get_model_files() else None
                )
                refresh_infer_model = gr.Button("刷新模型文件列表")
                infer_input = gr.Audio(label="上传音频文件", type="numpy")
                infer_output_path = gr.Textbox(
                    label="输出文件夹 (输出文件将基于输入文件名并添加_output后缀保存在outputs文件夹中)",
                    value="outputs")
                infer_key = gr.Number(label="变调 (半音数)", value=0)
                infer_speaker_id = gr.Number(label="目标说话人ID", value=1)
                infer_step = gr.Number(label="推理步数", value=10)
                infer_config = gr.Dropdown(
                    choices=[""] + get_exp_config_files(),
                    label="配置文件 (可选，从exp文件夹读取)",
                    value=""
                )
                refresh_infer_config = gr.Button("刷新配置文件列表")
                infer_button = gr.Button("开始推理")
            with gr.Column():
                infer_output = gr.Textbox(label="推理结果", lines=5, max_lines=10)
                infer_result = gr.Audio(label="推理结果音频")

    refresh_infer_model.click(
        lambda: gr.Dropdown(choices=get_model_files()),
        outputs=[infer_model]
    )

    refresh_infer_config.click(
        lambda: gr.Dropdown(choices=[""] + get_exp_config_files()),
        outputs=[infer_config]
    )

    infer_button.click(
        run_inference,
        inputs=[infer_model, infer_input, infer_output_path, infer_key, infer_speaker_id, infer_step, infer_config],
        outputs=[infer_output, infer_result]
    )

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1, max_size=20)
    app.launch(server_port=7860, server_name="0.0.0.0")
