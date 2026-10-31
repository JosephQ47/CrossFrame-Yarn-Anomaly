"""Start the local review service if needed, then open one reviewer's page."""

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(__file__).with_name("black_review_server.py")
BASE = "http://127.0.0.1:8765"


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/api/frames/A", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer", choices=("A", "B"))
    parser.add_argument("--no-open", action="store_true", help="health check without opening a browser")
    args = parser.parse_args()
    if not healthy():
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen([sys.executable, str(SERVER)], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags)
        for _ in range(40):
            if healthy():
                break
            time.sleep(.25)
        else:
            raise RuntimeError("审核服务启动失败；请运行 black_review_server.py 查看错误")
    url = BASE + "/?reviewer=" + args.reviewer
    if not args.no_open:
        webbrowser.open(url)
    print(url)


if __name__ == "__main__":
    main()
