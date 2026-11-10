import base64
from pathlib import Path

def base64_decode(data:str):
    return base64.b64decode(data.encode(encoding = 'utf-8'))

base64_txt_path = Path("./test/test_data/base64_error.txt")

with base64_txt_path.open("r") as f:
    audio_data = f.read()

base64_decode(audio_data)

    