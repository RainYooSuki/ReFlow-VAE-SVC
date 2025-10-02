#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorBoard监控脚本
用于监控exp文件夹内的训练日志数据
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path

def find_log_dirs(exp_dir="exp"):
    """
    查找exp目录下所有包含日志的子目录
    """
    log_dirs = []
    
    if not os.path.exists(exp_dir):
        print(f"目录 {exp_dir} 不存在")
        return log_dirs
    
    # 遍历exp目录下的所有子目录
    for root, dirs, files in os.walk(exp_dir):
        # 检查是否存在logs目录
        logs_path = os.path.join(root, "logs")
        if os.path.exists(logs_path) and os.path.isdir(logs_path):
            # 检查logs目录是否包含事件文件
            try:
                has_event_files = any(file.startswith("events.out.tfevents") for file in os.listdir(logs_path))
                if has_event_files:
                    # 使用相对路径作为日志标签
                    relative_path = os.path.relpath(root, exp_dir)
                    log_dirs.append((relative_path, logs_path))
            except OSError:
                # 忽略无法访问的目录
                continue
    
    return log_dirs

def start_tensorboard(log_dirs, port=6006):
    """
    启动TensorBoard监控指定的日志目录
    """
    if not log_dirs:
        print("未找到任何TensorBoard日志目录")
        return
    
    # 构建TensorBoard命令，添加 --host 0.0.0.0 参数
    cmd = ["tensorboard", "--port", str(port), "--host", "0.0.0.0"]
    
    # 如果只有一个日志目录，直接监控
    if len(log_dirs) == 1:
        log_dir_path = log_dirs[0][1]
        cmd.extend(["--logdir", log_dir_path])
        print(f"启动TensorBoard监控: {log_dirs[0][0]}")
    else:
        # 如果有多个日志目录，使用logdir_spec参数
        logdir_spec = ",".join([f"{name}:{path}" for name, path in log_dirs])
        cmd.extend(["--logdir_spec", logdir_spec])
        print("启动TensorBoard监控以下目录:")
        for name, path in log_dirs:
            print(f"  - {name}: {path}")
    
    print(f"TensorBoard将在 http://localhost:{port} 上可用")
    
    try:
        # 启动TensorBoard
        process = subprocess.Popen(cmd)
        # 打开浏览器
        webbrowser.open(f"http://localhost:{port}")
        # 等待进程结束
        process.wait()
    except KeyboardInterrupt:
        print("\n正在关闭TensorBoard...")
        process.terminate()
        process.wait()
        print("TensorBoard已关闭")
    except FileNotFoundError:
        print("错误: 未找到tensorboard命令，请确保已安装tensorboard")
        print("可以通过以下命令安装: pip install tensorboard")

def main():
    parser = argparse.ArgumentParser(description="TensorBoard监控exp文件夹内的训练日志")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="实验目录路径 (默认: exp)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="TensorBoard端口号 (默认: 6006)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出找到的日志目录而不启动TensorBoard"
    )
    
    args = parser.parse_args()
    
    # 查找日志目录
    log_dirs = find_log_dirs(args.exp_dir)
    
    if not log_dirs:
        print("在指定目录中未找到TensorBoard日志文件")
        return
    
    if args.list:
        print("找到以下TensorBoard日志目录:")
        for name, path in log_dirs:
            print(f"  - {name}: {path}")
        return
    
    # 启动TensorBoard
    start_tensorboard(log_dirs, args.port)

if __name__ == "__main__":
    main()