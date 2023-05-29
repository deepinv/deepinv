import torch
import os
import numpy as np


def get_freer_gpu():
    """
    Returns the GPU device with the most free memory.

    """
    try:
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
        memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        idx = np.argmax(memory_available)
        device = torch.device(f"cuda:{idx}")
        print(f"Selected GPU {idx} with {np.max(memory_available)} MB free memory ")
    except:
        device = torch.device(f"cuda")
        print("Couldn't find free GPU")

    return device


def save_model(epoch, model, optimizer, ckp_interval, epochs, loss, save_path):
    if (epoch > 0 and epoch % ckp_interval == 0) or epoch + 1 == epochs:
        os.makedirs(save_path, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, os.path.join(save_path, "ckp_{}.pth.tar".format(epoch)))
    pass


def load_checkpoint(model, path_checkpoint, device):
    checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def investigate_model(model, idx_max=1, check_name="iterator.g_step.g_param.0"):
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and (idx < idx_max or check_name in name):
            print(
                name,
                param.data.flatten()[0],
                "gradient norm = ",
                param.grad.detach().data.norm(2),
            )
