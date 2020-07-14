import os
import torch


def atomic_save(checkpoint, filepath):
    try:
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)
    except Exception as e:
        os.remove(tmp_path)
