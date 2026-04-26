"""
Setup script: creates a venv and installs all dependencies for the golf swing analyzer.
Run with: python setup.py
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    venv_dir = ROOT / ".venv"

    print("\n=== Golf Swing Analyzer Setup ===\n")

    print("[1/4] Creating virtual environment...")
    run([sys.executable, "-m", "venv", str(venv_dir)])

    pip = str(venv_dir / "bin" / "pip") if os.name != "nt" else str(venv_dir / "Scripts" / "pip")

    print("\n[2/4] Upgrading pip...")
    run([pip, "install", "--upgrade", "pip"])

    print("\n[3/4] Installing dependencies...")
    deps = [
        "mediapipe>=0.10.14",
        "opencv-python>=4.9",
        "numpy>=1.26",
        "scipy>=1.12",
        "tqdm>=4.66",
        "yt-dlp>=2024.3",
        "flask>=3.0",
        "flask-cors>=4.0",
        "anthropic>=0.25",
        "python-dotenv>=1.0",
    ]
    run([pip, "install"] + deps)

    print("\n[4/4] Creating .env.example...")
    env_example = ROOT / ".env.example"
    if not env_example.exists():
        env_example.write_text("ANTHROPIC_API_KEY=your_key_here\n")

    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file.write_text("ANTHROPIC_API_KEY=your_key_here\n")
        print("  Created .env — add your Anthropic API key before running api_server.py")

    (ROOT / "pro_swings").mkdir(exist_ok=True)

    python = str(venv_dir / "bin" / "python") if os.name != "nt" else str(venv_dir / "Scripts" / "python")

    print("\n=== Setup complete! ===")
    print("\nNext steps:")
    print(f"  1. Activate the venv:  source .venv/bin/activate")
    print(f"  2. Add your API key:   edit .env  (set ANTHROPIC_API_KEY=sk-ant-...)")
    print(f"  3. Download pro swings: python fetch_pro_swings.py")
    print(f"  4. Build reference model: python build_reference_model.py")
    print(f"  5. Start server:        python api_server.py")
    print(f"  6. Open browser:        http://localhost:8080\n")


if __name__ == "__main__":
    main()
