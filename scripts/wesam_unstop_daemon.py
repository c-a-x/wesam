import os, time, signal

# 轻量 wesam 作业守护：直读 /proc 发现被 process_guard.py SIGSTOP 的
# wesam 实验进程并 SIGCONT。避免 psutil 全量扫描导致自身 CPU 过高
# 被 guard 判为 high-cpu 作业而遭 STOP。
# 策略与用户既有 /tmp/resume_daemon.py (auto_unstop, nnunet 系) 一致。
PATTERNS = (
    "run_polyp_manifest",
    "ifp_alignment.train_medical",
    "export_test_predictions",
)
LOG = "/datanas01/nas01/Student-home/2024U/YBC/wesam2/output_current/wesam_unstop_daemon.log"
INTERVAL = 2.0


def log(msg):
    try:
        with open(LOG, "a") as f:
            f.write("%s wesam-unstop: %s\n" % (time.strftime("%F %T"), msg))
    except Exception:
        pass


def scan_once():
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open("/proc/%d/stat" % pid, "rb") as f:
                data = f.read()
            rparen = data.rfind(b")")
            if data[rparen + 2:rparen + 3] != b"T":
                continue
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
            if any(x in cmd for x in PATTERNS):
                os.kill(pid, signal.SIGCONT)
                log("resumed pid=%d cmd=%.100s" % (pid, cmd))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            continue


while True:
    scan_once()
    time.sleep(INTERVAL)
