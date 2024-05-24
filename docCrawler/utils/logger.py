import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(os.getcwd(), "logfile.log"),
    filemode="a",
)

info_logger = logging.getLogger(__name__) # Initialize the logger
