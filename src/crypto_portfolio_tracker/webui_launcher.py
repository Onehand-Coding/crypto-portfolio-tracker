import sys
import subprocess
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "webui.py"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(script_path),
                "--server.address", "0.0.0.0",
                "--server.port", "8502"
            ],
            check=True
        )
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"[x] Streamlit exited with error code {e.returncode}")
        sys.exit(e.returncode)
