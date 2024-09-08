
import torch
import cfg
from func_3d.utils import get_network
from func_3d.function import validation_sam
from func_3d.dataset import BTCV
from func_3d.dataset import get_dataloader
from hydra import initialize, compose
import hydra
from hydra.core.global_hydra import GlobalHydra

def main(args):


    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')

    nice_train_loader, nice_test_loader = get_dataloader(args)

    net = get_network(args, args.net, use_gpu=True, gpu_device=args.gpu_device)
 
    validation_loss, validation_metrics = validation_sam(args, nice_test_loader, 0, net)

    print(f"Validation Loss: {validation_loss}")
    print(f"Validation Metrics (IoU and Dice): {validation_metrics}")

if __name__ == "__main__":

    initialize(config_path=f"./sam2_train", version_base='1.2')
    args = cfg.parse_args() 
    main(args)
