import timm
import huggingface_hub as hf_hub
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    load_dotenv()

    HF_TOKEN = os.getenv('HF_TOKEN')

    hf_hub.login(token=HF_TOKEN)
    hf_hub.hf_hub_download('MahmoodLab/CONCH', 'pytorch_model.bin', local_dir='checkpoint/conch')
    hf_hub.hf_hub_download('MahmoodLab/UNI', 'pytorch_model.bin', local_dir='checkpoint/uni')
    hf_hub.hf_hub_download('MahmoodLab/UNI2-h', 'pytorch_model.bin', local_dir='checkpoint/uni_2')
