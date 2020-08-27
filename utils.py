import subprocess
import torch
import torch.utils.tensorboard as tf
import os


def use_tensorboard(print_dir):
    """returns process and instance of TensorBoard"""
    assert not torch.cuda.is_available(), 'TensorBoard not available on free GPUs on Paperspace Gradient'
    TB_process = subprocess.Popen(["tensorboard", f"--logdir={print_dir.parent}"], stdout=open(os.devnull, 'w'),
                                  stderr=subprocess.STDOUT)  # logdir={print_dir} to show just this run
    writer = tf.SummaryWriter(print_dir / 'TensorBoard_events')
    return TB_process, writer


def kill_tensorboard(TB_process):
    """kill TensorBoard"""
    subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
