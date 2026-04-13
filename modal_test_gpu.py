import modal

app = modal.App("test-gpu-access")

@app.function(gpu="T4", timeout=120)
def test_gpu():
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    return result

@app.local_entrypoint()
def main():
    print(test_gpu.remote())
