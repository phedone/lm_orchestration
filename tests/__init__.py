import logging
import sys

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)
test_logger.addHandler(console_handler)
