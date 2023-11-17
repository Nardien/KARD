from parallelformers import parallelize
import torch.nn as nn
import os

class parallelize2(parallelize):
    def __init__(self,
            model: nn.Module,
            fp16: bool,
            num_gpus: int,
            custom_policies=None,
            master_addr: str = "127.0.0.1",
            master_port: int = 29500,
            backend="nccl",
            verbose: str = None,
            init_method: str = "spawn",
            daemon: bool = True,
            seed: int = None,):
        super().__init__(model, fp16, num_gpus, custom_policies,
                         master_addr, master_port,
                         backend, verbose, init_method, daemon, seed)
    
    def init_environments(
        self,
        num_gpus: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        """
        Initialize environment variables
        Args:
            num_gpus (int): number of GPU for parallelization.
            master_addr (str): master process address for process communication
            master_port (int): master process port for process communication
        """
        print("PALLELFORMERS: Enter the new init environments")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(num_gpus)