"""Import this before training to suppress noisy dataloader warnings."""
import logging
import os

# Only show ERROR and above from root logger
level = os.environ.get("SAM2_LOG_LEVEL", "ERROR")
logging.getLogger().setLevel(getattr(logging, level, logging.ERROR))

# Specifically silence the noisy modules
for name in [
    "training.dataset.vos_raw_dataset",
    "training.dataset.vos_dataset",
    "training.dataset.vos_sampler",
]:
    logging.getLogger(name).setLevel(logging.ERROR)