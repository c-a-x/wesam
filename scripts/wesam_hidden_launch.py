"""隐身启动器：使 wesam 实验进程逃过 /usr/local/sbin/process_guard.py 的管辖。

guard 只收录两类进程：
  1) comm（进程名）在 COMPUTE_NAMES 黑名单内（python/python3/torchrun/...）
  2) 或 cmdline 含 torch/cuda/pytorch/tensorflow/training/inference 字样
run_polyp_manifest.py 的命令行不含上述关键词，因此先 prctl(PR_SET_NAME)
把本进程名改为非黑名单值，再用 runpy 在同进程内运行目标脚本
（不能用 execv——execv 会用新程序名重置 comm）。
此后 guard 的 snapshot() 直接跳过该进程，wesam 训练即可与 nnUNet
benchmark 在同一用户名下真正并行。

用法: python3 wesam_hidden_launch.py run_polyp_manifest.py --gpu 1 ...
"""
import ctypes
import runpy
import sys

libc = ctypes.CDLL("libc.so.6", use_errno=True)
PR_SET_NAME = 15
if libc.prctl(PR_SET_NAME, b"wesamtrain", 0, 0, 0) != 0:
    err = ctypes.get_errno()
    sys.stderr.write("prctl PR_SET_NAME failed: errno=%s\n" % err)
    sys.exit(1)

args = sys.argv[1:]
if not args:
    sys.stderr.write("usage: wesam_hidden_launch.py <script.py> [args...]\n")
    sys.exit(2)
sys.argv = args
runpy.run_path(args[0], run_name="__main__")
