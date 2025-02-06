import os
from utils import TBLogger


class TrainerBase(object):
    """
    Base class for all training class
    """
    def __init__(self, log_dir, sub_id=-1):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        self.ckpt_path = os.path.join("checkpoints", log_dir)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        if sub_id >= 0:
            self.tb_logger = TBLogger(log_path=self.ckpt_path, name=f"tensorboard_{sub_id}")

    def train_process(self):
        """
        This is the main function of a Trainer class
        Returns
        -------

        """
        raise NotImplementedError
