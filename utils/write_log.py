import os
project_index = os.getcwd().find('AdversialExamples')
root = os.getcwd()[0:project_index] + 'AdversialExamples'
import sys
sys.path.append(root)
import logging
from config.Config import opt




def log_here(level, context, path=opt.log_path, ifPrint=True):
    if ifPrint:
        print(context)
    assert level in ['info', 'debug']
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # keep print once
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if level == 'info':
        logger.info(context)
    elif level == 'debug':
        logger.debug(context)

