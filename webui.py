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
infer_process = None
preprocess_output_buffer = ""
train_output_buffer = ""
infer_output_buffer = ""
tensorboard_process = None  # 添加TensorBoard进程引用


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


def stop_inference():
    """
    停止音频推理
    """
    global infer_process
    if stop_process(infer_process):
        return "推理已停止"
    else:
        return "没有正在运行的推理任务"


def run_draw():
    """
    运行draw.py脚本
    """
    try:
        # 使用sys.executable确保使用当前Python环境
        cmd = [sys.executable, 'draw.py']

        # 执行draw.py脚本
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), encoding='utf-8')

        if result.returncode == 0:
            return f"draw.py执行成功！\n\n输出:\n{result.stdout}"
        else:
            return f"draw.py执行过程中出现错误:\n{result.stderr}"
    except Exception as e:
        return f"执行draw.py时出现错误: {str(e)}"


def run_preprocess(config_path, progress=gr.Progress()):
    """
    数据预处理功能
    """
    global preprocess_process, preprocess_output_buffer
    try:
        # 清空输出缓冲区
        preprocess_output_buffer = ""
        
        # 构建命令行参数
        cmd = [
            sys.executable, 'preprocess.py',
            '--config', config_path
        ]

        # 执行预处理脚本
        preprocess_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                              bufsize=1, universal_newlines=True, cwd=os.getcwd(), encoding='utf-8')
        
        # 实时读取输出
        while True:
            output = preprocess_process.stdout.readline()
            if output == '' and preprocess_process.poll() is not None:
                break
            if output:
                preprocess_output_buffer += output
                # 实时更新输出显示
                yield preprocess_output_buffer
        
        preprocess_process.wait()
        
        if preprocess_process.returncode == 0:
            final_output = f"预处理成功完成！\n\n输出:\n{preprocess_output_buffer}"
        else:
            final_output = f"预处理过程中出现错误:\n{preprocess_output_buffer}"
            
        yield final_output
    except Exception as e:
        yield f"执行预处理时出现错误: {str(e)}"


def stop_preprocess():
    """
    停止数据预处理
    """
    global preprocess_process
    if stop_process(preprocess_process):
        return "预处理已停止"
    else:
        return "没有正在运行的预处理任务"


def run_training(config_path, progress=gr.Progress()):
    """
    模型训练功能
    """
    global train_process, train_output_buffer
    try:
        # 清空输出缓冲区
        train_output_buffer = ""
        
        # 构建命令行参数
        cmd = [
            sys.executable, 'train.py',
            '--config', config_path
        ]

        # 执行训练脚本
        train_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         bufsize=1, universal_newlines=True, cwd=os.getcwd(), encoding='utf-8')
        
        # 实时读取输出
        while True:
            output = train_process.stdout.readline()
            if output == '' and train_process.poll() is not None:
                break
            if output:
                train_output_buffer += output
                # 实时更新输出显示
                yield train_output_buffer
        
        train_process.wait()
        
        if train_process.returncode == 0:
            final_output = f"训练启动成功！\n\n输出:\n{train_output_buffer}"
        else:
            final_output = f"训练启动过程中出现错误:\n{train_output_buffer}"
            
        yield final_output
    except Exception as e:
        yield f"执行训练时出现错误: {str(e)}"


def stop_training():
    """
    停止模型训练
    """
    global train_process
    if stop_process(train_process):
        return "训练已停止"
    else:
        return "没有正在运行的训练任务"


def run_inference(model_ckpt, input_file, input_folder, infer_mode, output_dir, key, speaker_id, infer_step, method, config_path, progress=gr.Progress()):
    """
    音频推理功能 - 支持单文件推理和批量推理
    """
    global infer_process, infer_output_buffer
    try:
        # 清空输出缓冲区
        infer_output_buffer = ""
        
        # 确保输出文件夹存在
        if not output_dir or output_dir == "outputs":
            output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        if infer_mode == "单文件推理":
            # 检查输入文件路径是否为空
            if not input_file or not os.path.exists(input_file):
                return "请提供有效的音频文件路径", None, None

            # 生成输出文件名：在原始文件名后添加"_output"后缀
            input_filename = os.path.basename(input_file)
            filename_without_ext, ext = os.path.splitext(input_filename)
            output_filename = f"{filename_without_ext}_output{ext}"
            output_path = os.path.join(output_dir, output_filename)

            # 直接使用推理方法（模型内部会处理切片）
            last_yielded = ""
            for output in run_audio_inference(model_ckpt, input_file, output_path, key, speaker_id, infer_step, method, config_path):
                if output != last_yielded:
                    last_yielded = output
                    yield output, None if "推理过程中出现错误" in output or "执行推理时出现错误" in output else output_path, None
        else:  # 批量推理
            # 检查输入文件夹是否存在
            if not os.path.exists(input_folder):
                yield f"输入文件夹 {input_folder} 不存在", None, None
                return
                
            # 获取所有音频文件
            audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
            audio_files = []
            for file in os.listdir(input_folder):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(file)
            
            if not audio_files:
                yield "输入文件夹中没有找到音频文件", None, None
                return
            
            # 存储所有输出文件路径
            output_files = []
            
            # 依次处理每个音频文件
            for i, audio_file in enumerate(audio_files):
                input_path = os.path.join(input_folder, audio_file)
                
                # 生成输出文件名
                filename_without_ext, ext = os.path.splitext(audio_file)
                output_filename = f"{filename_without_ext}_output{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                # 添加到输出文件列表
                output_files.append(output_path)
                
                # 更新进度信息
                progress_info = f"正在处理 ({i+1}/{len(audio_files)}): {audio_file}\n"
                yield progress_info, None, None
                
                # 执行推理
                for output in run_audio_inference(model_ckpt, input_path, output_path, key, speaker_id, infer_step, method, config_path):
                    # 实时显示推理过程
                    combined_output = progress_info + output
                    yield combined_output, None, None
            
            # 所有文件处理完成后
            success_msg = f"批量推理完成！共处理 {len(audio_files)} 个文件。\n输出文件列表：\n" + "\n".join(output_files)
            yield success_msg, None, output_files
            
    except Exception as e:
        # 添加更详细的错误信息
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"执行推理时出现错误: {str(e)}\n详细信息:\n{error_details}"
        yield error_msg, None, None


def run_audio_inference(model_ckpt, input_path, output_path, key, speaker_id, infer_step, method, config_path):
    """
    音频推理方法 - 实时输出版本
    """
    global infer_process, infer_output_buffer
    try:
        # 构建命令行参数，按照 python main.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> 
        # -k <keychange (semitones)> -tid <target_speaker_id> -step <infer_step> -method <method> 格式
        cmd = [
            sys.executable, 'main.py',
            '-i', input_path,
            '-m', model_ckpt,
            '-o', output_path,
            '-k', str(key),
            '-tid', str(speaker_id),
            '-step', str(infer_step),
            '-method', method
        ]

        # 执行推理脚本并实时输出
        infer_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   cwd=os.getcwd(), encoding='utf-8')
        
        last_yielded = ""
        # 实时读取输出
        while True:
            output = infer_process.stdout.readline()
            if output == '' and infer_process.poll() is not None:
                break
            if output:
                infer_output_buffer += output
                # 实时更新输出显示（仅在有新内容时yield）
                if infer_output_buffer != last_yielded:
                    last_yielded = infer_output_buffer
                    yield infer_output_buffer
        
        infer_process.wait()

        if infer_process.returncode == 0:
            if os.path.exists(output_path):
                final_output = f"推理成功完成！\n\n输出日志:\n{infer_output_buffer}"
            else:
                final_output = f"推理完成但未找到输出文件:\n\n输出日志:\n{infer_output_buffer}"
        else:
            final_output = f"推理过程中出现错误:\n\n输出日志:\n{infer_output_buffer}"
            
        if final_output != last_yielded:
            yield final_output
    except Exception as e:
        # 添加更详细的错误信息
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"执行推理时出现错误: {str(e)}\n详细信息:\n{error_details}"
        if error_msg != last_yielded:
            yield error_msg


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


def get_exp_config_yaml_files():
    """
    获取exp文件夹中的config.yaml文件列表
    """
    exp_dir = "exp"
    config_files = []
    if os.path.exists(exp_dir):
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file == 'config.yaml':
                    config_files.append(os.path.join(root, file))
    return config_files


def start_tensorboard(port=6006):
    """
    启动TensorBoard监控exp目录下的日志
    """
    global tensorboard_process
    
    try:
        import subprocess
        import webbrowser
        import os
        import time
        import threading
        
        # 检查是否已经有TensorBoard进程在运行
        if tensorboard_process is not None and tensorboard_process.poll() is None:
            tensorboard_url = f"http://localhost:{port}"
            webbrowser.open(tensorboard_url)
            return f"TensorBoard已在运行中\n请在浏览器中查看: {tensorboard_url}"
        
        # 检查exp目录是否存在
        if not os.path.exists("exp"):
            return "错误: exp目录不存在"
        
        # 查找所有包含日志的子目录
        log_dirs = []
        for root, dirs, files in os.walk("exp"):
            logs_path = os.path.join(root, "logs")
            if os.path.exists(logs_path) and os.path.isdir(logs_path):
                # 检查logs目录是否包含事件文件
                try:
                    has_event_files = any(file.startswith("events.out.tfevents") for file in os.listdir(logs_path))
                    if has_event_files:
                        # 使用相对路径作为日志标签
                        relative_path = os.path.relpath(root, "exp")
                        if relative_path == ".":
                            name = os.path.basename(root)
                        else:
                            name = relative_path
                        log_dirs.append(f"{name}:{logs_path}")
                except OSError:
                    # 忽略无法访问的目录
                    continue
        
        if not log_dirs:
            return "错误: 在exp目录中未找到TensorBoard日志文件"
        
        # 检查端口是否被占用，如果被占用则寻找新的端口
        original_port = port
        if is_port_in_use(port):
            port = find_free_port()
        
        # 构建TensorBoard命令
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--port", str(port),
            "--host", "0.0.0.0"  # 允许外部访问
        ]
        
        # 如果只有一个日志目录
        if len(log_dirs) == 1:
            cmd.extend(["--logdir", log_dirs[0].split(":", 1)[1]])
        else:
            # 如果有多个日志目录
            cmd.extend(["--logdir_spec", ",".join(log_dirs)])
        
        # 使用独立线程启动TensorBoard
        def run_tensorboard():
            global tensorboard_process
            try:
                # 启动TensorBoard进程
                tensorboard_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 等待一点时间确保TensorBoard启动
                time.sleep(5)
                
                # 检查进程是否仍在运行
                if tensorboard_process.poll() is None:
                    print(f"TensorBoard已在端口 {port} 启动")
                else:
                    # 读取错误输出
                    stdout, stderr = tensorboard_process.communicate()
                    error_msg = stderr.decode('utf-8') if stderr else "未知错误"
                    print(f"启动TensorBoard时出错:\n{error_msg}")
            except Exception as e:
                print(f"启动TensorBoard时出现异常: {str(e)}")
        
        # 在独立线程中启动TensorBoard
        thread = threading.Thread(target=run_tensorboard, daemon=True)
        thread.start()
        
        # 等待一小段时间让TensorBoard启动
        time.sleep(2)
        
        # 打开浏览器
        tensorboard_url = f"http://localhost:{port}"
        webbrowser.open(tensorboard_url)
        
        port_info = f"端口 {original_port} 已被占用，使用端口 {port}" if port != original_port else f"端口 {port}"
        
        if len(log_dirs) == 1:
            return f"TensorBoard启动命令已发送，{port_info}\n监控目录: {log_dirs[0].split(':', 1)[0]}\n请在浏览器中查看: {tensorboard_url}"
        else:
            dirs_list = "\n".join([f"  - {dir_name}" for dir_name in [d.split(':', 1)[0] for d in log_dirs]])
            return f"TensorBoard启动命令已发送，{port_info}\n监控以下目录:\n{dirs_list}\n请在浏览器中查看: {tensorboard_url}"
    except Exception as e:
        import traceback
        return f"启动TensorBoard时出现异常: {str(e)}\n{traceback.format_exc()}"


def stop_tensorboard():
    """
    停止TensorBoard进程
    """
    global tensorboard_process
    try:
        import subprocess
        import os
        
        # 如果有TensorBoard进程在运行，则终止它
        if tensorboard_process is not None and tensorboard_process.poll() is None:
            tensorboard_process.terminate()
            tensorboard_process.wait()
            tensorboard_process = None
            return "TensorBoard已停止"
        
        # 查找并终止TensorBoard进程
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/f", "/im", "tensorboard.exe"], capture_output=True)
        else:  # Unix/Linux/Mac
            subprocess.run(["pkill", "-f", "tensorboard"], capture_output=True)
        
        return "TensorBoard已停止"
    except Exception as e:
        return f"停止TensorBoard时出现异常: {str(e)}"


def is_port_in_use(port):
    """
    检查端口是否被占用
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True


def find_free_port():
    """
    查找可用的端口
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


with gr.Blocks(title="ReFlow VAE SVC WebUI") as app:
    gr.Markdown("# ReFlow VAE SVC WebUI")
    gr.Markdown("一个基于Gradio的图形界面，用于音频预处理、训练和推理")
    
    with gr.Tab("TensorBoard监控"):
        with gr.Row():
            with gr.Column():
                tensorboard_port = gr.Number(label="TensorBoard端口", value=6006)
                start_tensorboard_button = gr.Button("启动TensorBoard")
                stop_tensorboard_button = gr.Button("停止TensorBoard")
            with gr.Column():
                tensorboard_output = gr.Textbox(label="TensorBoard状态", lines=5, max_lines=10)
    
    start_tensorboard_button.click(
        start_tensorboard,
        inputs=[tensorboard_port],
        outputs=[tensorboard_output]
    )
    
    stop_tensorboard_button.click(
        stop_tensorboard,
        outputs=[tensorboard_output]
    )

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
                
                def get_available_encoders():
                    """
                    获取可用的编码器选项
                    """
                    # 基本编码器选项 - 确保包含所有指定的编码器选项
                    encoders = [
                        'hubertsoft', 'hubertbase', 'hubertbase768', 
                        'contentvec', 'contentvec768', 'contentvec768l12',
                        'cnhubertsoftfish', 'whisper-ppg', 'whisper-ppg-large'
                    ]
                    
                    # 检查pretrain目录中的实际文件
                    pretrain_path = "pretrain"
                    
                    if os.path.exists(pretrain_path):
                        # 检查各个编码器的预训练文件是否存在
                        available_encoders = []
                        
                        # 检查contentvec相关文件
                        if os.path.exists(os.path.join(pretrain_path, "contentvec")):
                            available_encoders.extend(['contentvec', 'contentvec768', 'contentvec768l12'])
                        
                        # 检查whisper相关文件
                        if os.path.exists(os.path.join(pretrain_path, "whisper-ppg-large", "large-v2.pt")):
                            available_encoders.extend(['whisper-ppg', 'whisper-ppg-large'])
                        
                        # 检查hubert相关文件
                        if (os.path.exists(os.path.join(pretrain_path, "hubert", "hubert_soft.pt")) or
                            os.path.exists(os.path.join(pretrain_path, "hubert", "hubert_base.pt")) or
                            os.path.exists(os.path.join(pretrain_path, "hubert", "hubert_base_768.pt"))):
                            available_encoders.extend(['hubertsoft', 'hubertbase', 'hubertbase768'])
                        
                        # 检查cnhubertsoftfish相关文件
                        if os.path.exists(os.path.join(pretrain_path, "cnhubertsoftfish")):
                            available_encoders.append('cnhubertsoftfish')
                        
                        # 如果有可用的编码器文件，则返回基本编码器和可用编码器的交集
                        # 但确保所有基本编码器都在列表中（即使文件不存在）
                        result = list(set(encoders) & set(available_encoders)) if available_encoders else encoders
                        # 确保所有基本编码器都在结果中
                        for encoder in encoders:
                            if encoder not in result:
                                result.append(encoder)
                        return result
                    else:
                        # 如果pretrain目录不存在，返回所有基本编码器选项
                        return encoders
                
                def get_encoder_info(encoder_name):
                    """
                    获取编码器信息（路径和输出通道数）
                    """
                    encoder_paths = {
                        'hubertsoft': 'pretrain/hubert/hubert_soft.pt',
                        'hubertbase': 'pretrain/hubert/hubert_base.pt',
                        'hubertbase768': 'pretrain/hubert/hubert_base_768.pt',
                        'contentvec': 'pretrain/contentvec/checkpoint_best_legacy_500.pt',
                        'contentvec768': 'pretrain/contentvec/checkpoint_best_legacy_500.pt',
                        'contentvec768l12': 'pretrain/contentvec/checkpoint_best_legacy_500.pt',
                        'cnhubertsoftfish': 'pretrain/cnhubertsoftfish/checkpoint_best_legacy_500.pt',
                        'whisper-ppg': 'pretrain/whisper-ppg-large/large-v2.pt',
                        'whisper-ppg-large': 'pretrain/whisper-ppg-large/large-v2.pt'
                    }
                    
                    encoder_channels = {
                        'hubertsoft': 256,
                        'hubertbase': 256,
                        'hubertbase768': 768,
                        'contentvec': 768,
                        'contentvec768': 768,
                        'contentvec768l12': 768,
                        'cnhubertsoftfish': 256,
                        'whisper-ppg': 1024,
                        'whisper-ppg-large': 1280
                    }
                    
                    path = encoder_paths.get(encoder_name, '未知路径')
                    channels = encoder_channels.get(encoder_name, '未知')
                    
                    # 检查路径是否存在
                    if path != '未知路径' and os.path.exists(path):
                        path_status = f"{path} (存在)"
                    elif path != '未知路径':
                        path_status = f"{path} (不存在)"
                    else:
                        path_status = path
                    
                    return path, path_status, channels  # 返回实际路径、显示路径和通道数
                
                def update_encoder_info(encoder_name):
                    """
                    更新编码器信息显示
                    """
                    path, path_status, channels = get_encoder_info(encoder_name)
                    return [
                        gr.update(value=path),  # 实际路径用于保存
                        gr.update(value=path_status),  # 显示路径包含存在状态
                        gr.update(value=channels)
                    ]
                
                # Encoder选择下拉框
                preprocess_encoder = gr.Dropdown(
                    choices=get_available_encoders(),
                    label="选择编码器 (Encoder)",
                    value="contentvec768l12"  # 默认值与配置文件保持一致
                )
                
                # 隐藏的实际路径字段
                preprocess_encoder_path = gr.Textbox(
                    label="编码器路径",
                    value="pretrain/contentvec/checkpoint_best_legacy_500.pt",
                    interactive=False,
                    visible=False  # 隐藏该字段
                )
                
                # 显示用的路径字段
                preprocess_encoder_path_display = gr.Textbox(
                    label="编码器路径",
                    value="pretrain/contentvec/checkpoint_best_legacy_500.pt (存在)",
                    interactive=False
                )
                
                preprocess_encoder_channels = gr.Number(
                    label="编码器输出通道数 (encoder_out_channels)",
                    value=768,
                    interactive=False
                )
                
                # 添加刷新按钮
                refresh_encoder_list = gr.Button("刷新编码器列表")
                
                refresh_preprocess_config = gr.Button("刷新配置文件列表")
                load_preprocess_config_button = gr.Button("加载预处理配置")
                save_preprocess_config_button = gr.Button("保存预处理配置")
                preprocess_button = gr.Button("开始预处理")
                stop_preprocess_button = gr.Button("停止预处理")
            with gr.Column():
                preprocess_output = gr.Textbox(label="预处理输出", lines=10, max_lines=20)


    def load_preprocess_config(config_path):
        """
        加载预处理配置
        """
        config_data = load_config_data(config_path)
        if config_data is None:
            return [gr.update(), gr.update(), gr.update(), gr.update()]
        
        # 获取数据参数
        data_params = config_data.get('data', {})
        encoder = data_params.get('encoder', 'contentvec768l12')
        
        # 获取编码器路径和通道数
        encoder_ckpt = data_params.get('encoder_ckpt', '')
        encoder_out_channels = data_params.get('encoder_out_channels', 768)
        
        # 构造显示用的路径
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            encoder_ckpt_display = f"{encoder_ckpt} (存在)"
        elif encoder_ckpt:
            encoder_ckpt_display = f"{encoder_ckpt} (不存在)"
        else:
            encoder_ckpt_display = "未知路径"
        
        return [
            gr.update(value=encoder),
            gr.update(value=encoder_ckpt),
            gr.update(value=encoder_ckpt_display),
            gr.update(value=encoder_out_channels)
        ]


    def save_preprocess_config(config_path, encoder, encoder_path, encoder_channels):
        """
        保存预处理配置
        """
        config_data = load_config_data(config_path)
        if config_data is None:
            return "加载配置文件失败"
        
        # 更新数据参数
        if 'data' not in config_data:
            config_data['data'] = {}
            
        config_data['data']['encoder'] = encoder
        config_data['data']['encoder_ckpt'] = encoder_path  # 保存实际路径，不包含状态文本
        config_data['data']['encoder_out_channels'] = int(encoder_channels)
        
        # 保存配置文件
        if save_config_data(config_path, config_data):
            return f"预处理配置已保存到 {config_path}"
        else:
            return "保存配置文件失败"


    def update_preprocess_output():
        global preprocess_output_buffer
        return preprocess_output_buffer


    draw_button.click(
        run_draw,
        outputs=[preprocess_output]
    )

    refresh_preprocess_config.click(
        lambda: gr.Dropdown(choices=get_config_files()),
        outputs=[preprocess_config]
    )
    
    load_preprocess_config_button.click(
        load_preprocess_config,
        inputs=[preprocess_config],
        outputs=[preprocess_encoder, preprocess_encoder_path, preprocess_encoder_path_display, preprocess_encoder_channels]
    )
    
    save_preprocess_config_button.click(
        save_preprocess_config,
        inputs=[preprocess_config, preprocess_encoder, preprocess_encoder_path, preprocess_encoder_channels],
        outputs=[preprocess_output]
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
    
    # 添加编码器选择变化事件
    preprocess_encoder.change(
        update_encoder_info,
        inputs=[preprocess_encoder],
        outputs=[preprocess_encoder_path, preprocess_encoder_path_display, preprocess_encoder_channels]
    )
    
    # 初始化时更新编码器信息
    app.load(
        update_encoder_info,
        inputs=[preprocess_encoder],
        outputs=[preprocess_encoder_path, preprocess_encoder_path_display, preprocess_encoder_channels]
    )
    
    with gr.Tab("模型训练"):
        with gr.Row():
            with gr.Column():
                # 添加继续训练开关
                train_resume_checkbox = gr.Checkbox(label="继续训练", value=False)
                
                # 原始配置文件选择（继续训练关闭时显示）
                train_config = gr.Dropdown(
                    choices=get_config_files(),
                    label="选择配置文件",
                    value=get_config_files()[0] if get_config_files() else None
                )
                
                # 继续训练时的配置文件选择（继续训练开启时显示）
                train_resume_config = gr.Dropdown(
                    choices=get_exp_config_yaml_files(),  # 只显示config.yaml文件
                    label="选择配置文件 (继续训练)",
                    value=get_exp_config_yaml_files()[0] if get_exp_config_yaml_files() else None,
                    visible=False  # 默认隐藏
                )
                
                # 继续训练时的模型文件选择
                train_resume_model = gr.Dropdown(
                    choices=get_model_files(),
                    label="选择模型文件 (继续训练)",
                    value=get_model_files()[0] if get_model_files() else None,
                    visible=False  # 默认隐藏
                )
                
                with gr.Row():
                    refresh_train_config = gr.Button("刷新配置文件列表")
                    refresh_train_resume_config = gr.Button("刷新继续训练配置文件列表", visible=False)
                    refresh_train_resume_model = gr.Button("刷新模型文件列表", visible=False)
                
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
                
                # 新增的训练参数
                train_amp_dtype = gr.Dropdown(
                    label="amp_dtype", 
                    choices=["fp32", "fp16", "bf16"],
                    value="fp32"
                )
                train_interval_force_save = gr.Number(label="interval_force_save", value=5000)
                train_interval_log = gr.Number(label="interval_log", value=200)
                train_interval_val = gr.Number(label="interval_val", value=2000)
                train_save_opt = gr.Checkbox(label="save_opt", value=True)

                with gr.Row():
                    load_config_button = gr.Button("加载配置")
                    save_config_button = gr.Button("保存配置")
                
                with gr.Row():
                    load_resume_config_button = gr.Button("加载配置 (继续训练)", visible=False)
                    save_resume_config_button = gr.Button("保存配置 (继续训练)", visible=False)
                
                with gr.Row():
                    train_button = gr.Button("开始训练")
                    resume_train_button = gr.Button("继续训练", visible=False)  # 新增继续训练按钮，默认隐藏
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
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]

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
        
        # 新增的训练参数
        amp_dtype = train_params.get('amp_dtype', 'fp32')
        interval_force_save = train_params.get('interval_force_save', 5000)
        interval_log = train_params.get('interval_log', 200)
        interval_val = train_params.get('interval_val', 2000)
        save_opt = train_params.get('save_opt', True)

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
            gr.update(value=weight_decay),
            gr.update(value=amp_dtype),
            gr.update(value=interval_force_save),
            gr.update(value=interval_log),
            gr.update(value=interval_val),
            gr.update(value=save_opt)
        ]


    def load_resume_train_config(config_path):
        """
        加载继续训练配置
        """
        # 继续训练使用相同的加载逻辑
        return load_train_config(config_path)


    def save_train_config(config_path, n_layers, n_chans, n_hidden, num_workers, batch_size, epochs, lr, decay_step,
                          gamma, weight_decay, amp_dtype, interval_force_save, interval_log, interval_val, save_opt):
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
        
        # 保存新增的训练参数
        config_data['train']['amp_dtype'] = amp_dtype
        config_data['train']['interval_force_save'] = int(interval_force_save)
        config_data['train']['interval_log'] = int(interval_log)
        config_data['train']['interval_val'] = int(interval_val)
        config_data['train']['save_opt'] = bool(save_opt)

        # 保存配置文件
        if save_config_data(config_path, config_data):
            return f"配置已保存到 {config_path}"
        else:
            return "保存配置文件失败"


    def save_resume_train_config(config_path, n_layers, n_chans, n_hidden, num_workers, batch_size, epochs, lr, decay_step,
                                 gamma, weight_decay, amp_dtype, interval_force_save, interval_log, interval_val, save_opt):
        """
        保存继续训练配置
        """
        # 继续训练使用相同的保存逻辑
        return save_train_config(config_path, n_layers, n_chans, n_hidden, num_workers, batch_size, epochs, lr, decay_step,
                                 gamma, weight_decay, amp_dtype, interval_force_save, interval_log, interval_val, save_opt)

    # 切换继续训练模式的函数
    def toggle_resume_mode(resume_enabled):
        """
        切换继续训练模式
        """
        if resume_enabled:
            # 继续训练模式：显示继续训练的配置和模型选择，隐藏原始配置选择
            return [
                gr.update(visible=False),  # train_config
                gr.update(visible=True),   # train_resume_config
                gr.update(visible=True),   # train_resume_model
                gr.update(visible=False),  # refresh_train_config
                gr.update(visible=True),   # refresh_train_resume_config
                gr.update(visible=True),   # refresh_train_resume_model
                gr.update(visible=False),  # load_config_button (原始)
                gr.update(visible=False),  # save_config_button (原始)
                gr.update(visible=True),   # load_resume_config_button
                gr.update(visible=True),   # save_resume_config_button
                gr.update(visible=False),  # train_button
                gr.update(visible=True)    # resume_train_button
            ]
        else:
            # 正常训练模式：显示原始配置选择，隐藏继续训练的配置和模型选择
            return [
                gr.update(visible=True),   # train_config
                gr.update(visible=False),  # train_resume_config
                gr.update(visible=False),  # train_resume_model
                gr.update(visible=True),   # refresh_train_config
                gr.update(visible=False),  # refresh_train_resume_config
                gr.update(visible=False),  # refresh_train_resume_model
                gr.update(visible=True),   # load_config_button (原始)
                gr.update(visible=True),   # save_config_button (原始)
                gr.update(visible=False),  # load_resume_config_button
                gr.update(visible=False),  # save_resume_config_button
                gr.update(visible=True),   # train_button
                gr.update(visible=False)   # resume_train_button
            ]

    # 绑定继续训练开关事件
    train_resume_checkbox.change(
        toggle_resume_mode,
        inputs=[train_resume_checkbox],
        outputs=[
            train_config, 
            train_resume_config, 
            train_resume_model,
            refresh_train_config,
            refresh_train_resume_config,
            refresh_train_resume_model,
            load_config_button,
            save_config_button,
            load_resume_config_button,
            save_resume_config_button,
            train_button,
            resume_train_button
        ]
    )

    refresh_train_config.click(
        lambda: gr.Dropdown(choices=get_config_files()),
        outputs=[train_config]
    )
    
    refresh_train_resume_config.click(
        lambda: gr.Dropdown(choices=get_exp_config_yaml_files()),  # 只显示config.yaml文件
        outputs=[train_resume_config]
    )
    
    refresh_train_resume_model.click(
        lambda: gr.Dropdown(choices=get_model_files()),
        outputs=[train_resume_model]
    )

    load_config_button.click(
        load_train_config,
        inputs=[train_config],
        outputs=[train_n_layers, train_n_chans, train_n_hidden, train_num_workers, train_batch_size, train_epochs,
                 train_lr, train_decay_step, train_gamma, train_weight_decay, train_amp_dtype,
                 train_interval_force_save, train_interval_log, train_interval_val, train_save_opt]
    )

    save_config_button.click(
        save_train_config,
        inputs=[train_config, train_n_layers, train_n_chans, train_n_hidden, train_num_workers, train_batch_size,
                train_epochs, train_lr, train_decay_step, train_gamma, train_weight_decay,
                train_amp_dtype, train_interval_force_save, train_interval_log, train_interval_val, train_save_opt],
        outputs=[train_output]
    )
    
    # 添加继续训练模式下的按钮点击事件
    load_resume_config_button.click(
        load_resume_train_config,
        inputs=[train_resume_config],
        outputs=[train_n_layers, train_n_chans, train_n_hidden, train_num_workers, train_batch_size, train_epochs,
                 train_lr, train_decay_step, train_gamma, train_weight_decay, train_amp_dtype,
                 train_interval_force_save, train_interval_log, train_interval_val, train_save_opt]
    )

    save_resume_config_button.click(
        save_resume_train_config,
        inputs=[train_resume_config, train_n_layers, train_n_chans, train_n_hidden, train_num_workers, train_batch_size,
                train_epochs, train_lr, train_decay_step, train_gamma, train_weight_decay,
                train_amp_dtype, train_interval_force_save, train_interval_log, train_interval_val, train_save_opt],
        outputs=[train_output]
    )

    train_button.click(
        run_training,
        inputs=[train_config],
        outputs=[train_output]
    )
    
    # 添加继续训练按钮的点击事件
    def run_resume_training(resume_config_path, resume_model_path):
        """
        运行继续训练
        """
        global train_process, train_output_buffer
        try:
            # 清空输出缓冲区
            train_output_buffer = ""
            
            # 构建命令行参数
            cmd = [
                sys.executable, 'train.py',
                '--config', resume_config_path
            ]
            
            # 如果指定了模型路径，则添加模型路径参数
            if resume_model_path and os.path.exists(resume_model_path):
                cmd.extend(['--model', resume_model_path])

            # 执行训练脚本
            train_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                             bufsize=1, universal_newlines=True, cwd=os.getcwd(), encoding='utf-8')
            
            # 实时读取输出
            while True:
                output = train_process.stdout.readline()
                if output == '' and train_process.poll() is not None:
                    break
                if output:
                    train_output_buffer += output
                    # 实时更新输出显示
                    yield train_output_buffer
            
            train_process.wait()
            
            if train_process.returncode == 0:
                final_output = f"训练启动成功！\n\n输出:\n{train_output_buffer}"
            else:
                final_output = f"训练启动过程中出现错误:\n{train_output_buffer}"
                
            yield final_output
        except Exception as e:
            yield f"执行训练时出现错误: {str(e)}"

    resume_train_button.click(
        run_resume_training,
        inputs=[train_resume_config, train_resume_model],
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
                
                # 添加推理模式选择
                infer_mode = gr.Radio(
                    choices=["单文件推理", "批量推理"],
                    label="推理模式",
                    value="单文件推理"
                )
                
                # 单文件推理组件
                infer_input_file = gr.Textbox(
                    label="输入文件路径 (可通过拖拽文件到下方区域获取路径)",
                    value=""
                )
                infer_input_file_upload = gr.File(label="拖拽文件到此处获取路径", file_types=["audio"])
                
                # 批量推理组件
                infer_input_folder = gr.Textbox(
                    label="输入文件夹路径 (包含需要推理的音频文件)",
                    value="input",
                    visible=False  # 默认隐藏，只有在批量推理模式下才显示
                )
                infer_output_path = gr.Textbox(
                    label="输出文件夹 (输出文件将基于输入文件名并添加_output后缀保存在outputs文件夹中)",
                    value="outputs")
                infer_key = gr.Number(label="变调 (半音数)", value=0)
                infer_speaker_id = gr.Number(label="目标说话人ID", value=1)
                infer_step = gr.Number(label="推理步数", value=10)
                infer_method = gr.Dropdown(
                    choices=["auto", "euler", "rk4"],
                    label="推理方法",
                    value="auto"
                )
                infer_config = gr.Dropdown(
                    choices=[""] + get_exp_config_files(),
                    label="配置文件 (可选，从exp文件夹读取)",
                    value=""
                )
                refresh_infer_config = gr.Button("刷新配置文件列表")
                with gr.Row():
                    infer_button = gr.Button("开始推理")
                    stop_infer_button = gr.Button("停止推理")
            with gr.Column():
                infer_output = gr.Textbox(label="推理结果", lines=5, max_lines=10)
                infer_result = gr.Audio(label="推理结果音频 (单文件模式下显示)")
                
                # 批量推理结果显示
                infer_result_gallery = gr.Gallery(
                    label="推理结果音频 (批量模式下显示)",
                    visible=False
                )
    
    refresh_infer_model.click(
        lambda: gr.Dropdown(choices=get_model_files()),
        outputs=[infer_model]
    )

    refresh_infer_config.click(
        lambda: gr.Dropdown(choices=[""] + get_exp_config_files()),
        outputs=[infer_config]
    )
    
    # 添加推理模式切换功能
    def toggle_infer_mode(mode):
        """
        切换推理模式时显示/隐藏相应组件
        """
        if mode == "单文件推理":
            return [
                gr.update(visible=True),   # infer_input_file (单文件路径)
                gr.update(visible=True),   # infer_input_file_upload (单文件上传用于获取路径)
                gr.update(visible=False),  # infer_input_folder (输入文件夹)
                gr.update(visible=True),   # infer_result (单文件结果)
                gr.update(visible=False)   # infer_result_gallery (批量结果)
            ]
        else:  # 批量推理
            return [
                gr.update(visible=False),  # infer_input_file (单文件路径)
                gr.update(visible=False),  # infer_input_file_upload (单文件上传用于获取路径)
                gr.update(visible=True),   # infer_input_folder (输入文件夹)
                gr.update(visible=False),  # infer_result (单文件结果)
                gr.update(visible=True)    # infer_result_gallery (批量结果)
            ]
    
    infer_mode.change(
        toggle_infer_mode,
        inputs=[infer_mode],
        outputs=[infer_input_file, infer_input_file_upload, infer_input_folder, infer_result, infer_result_gallery]
    )
    
    # 添加文件上传事件处理函数，用于获取文件路径
    def update_input_file_path(file_upload):
        """
        通过上传的文件更新输入文件路径
        """
        if file_upload is not None:
            return file_upload.name
        return ""
    
    infer_input_file_upload.change(
        update_input_file_path,
        inputs=[infer_input_file_upload],
        outputs=[infer_input_file]
    )

    infer_button.click(
        run_inference,
        inputs=[infer_model, infer_input_file, infer_input_folder, infer_mode, infer_output_path, infer_key, infer_speaker_id, infer_step, infer_method, infer_config],
        outputs=[infer_output, infer_result, infer_result_gallery]
    )
    
    stop_infer_button.click(
        stop_inference,
        outputs=[infer_output]
    )

    # 添加事件处理
    preprocess_encoder.change(
        update_encoder_info,
        inputs=[preprocess_encoder],
        outputs=[preprocess_encoder_path, preprocess_encoder_path_display, preprocess_encoder_channels]
    )
    
    refresh_encoder_list.click(
        lambda: gr.Dropdown(choices=get_available_encoders()),
        outputs=[preprocess_encoder]
    )
    
    # 初始化时更新编码器信息
    app.load(
        update_encoder_info,
        inputs=[preprocess_encoder],
        outputs=[preprocess_encoder_path, preprocess_encoder_path_display, preprocess_encoder_channels]
    )

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1, max_size=20)
    app.launch(server_port=7860, server_name="0.0.0.0")
