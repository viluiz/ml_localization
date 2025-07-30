"""
Allow running the module using -> python -m <module name>

@author: Vinicius Luiz Santos Silva

"""
from .cli import main
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()