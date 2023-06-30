from ..config import get_config
from ..utils.logger import Logger

logger = Logger.get_logger("DATASET")
CFG_PATH = 'src/config/config.yml'
CFG = get_config(CFG_PATH)