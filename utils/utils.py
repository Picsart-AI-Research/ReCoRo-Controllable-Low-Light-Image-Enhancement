from time import time
import torch
from PIL import ImageFont
from PIL import ImageDraw 

def get_device(opt):
    if torch.cuda.is_available() and opt.ngpu > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    return torch.device(device)


def timeit(name):
    def decorator(function):
        def wrapper(*args, **kwargs):
            start = time()
            result = function(*args, **kwargs)
            end = time()
            print(f"{name}: took: {end - start:2.4f} sec")
            return result
        return wrapper
    return decorator


