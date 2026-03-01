import logging
from logging.handlers import RotatingFileHandler


class TrainingLogger:
    """Logger utility for training."""

    def __init__(self, 
                 log_path: str = './logs/training.log', 
                 level: int = logging.INFO,
                 max_log_size: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        self.logger = logging.getLogger('ASRTrainer')
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        file_handler = RotatingFileHandler(log_path, maxBytes=max_log_size, backupCount=backup_count)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def log_training_resume(self, epoch: int, global_step: int, total_epochs: int):
        resume_message = (
            f"Training Resumed:\n"
            f"   Current Epoch: {epoch}\n"
            f"   Global Step: {global_step}\n"
            f"   Total Epochs: {total_epochs}\n"
            f"   Remaining Epochs: {total_epochs - epoch}"
        )
        self.info(resume_message)
