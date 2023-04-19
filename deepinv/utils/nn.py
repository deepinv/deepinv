import torch
import os


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
