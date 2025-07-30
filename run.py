"""
Allow running the module using -> python run.py

@author: Vinicius Luiz Santos Silva

"""
from ml_localization.cli import main
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    main()