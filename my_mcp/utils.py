import logging
import json
import sys

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("my_mcp")
logger.setLevel(logging.CRITICAL)

def load_config_from_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                print(f"Error: Unexpected configuration file extension for {file_path}. Please use .json")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        sys.exit(1)

# 处理deepseek-r1的reasoning_content
def clean_reasoning_content(conversation):
    for message in conversation:
        if 'reasoning_content' in message:
            del message['reasoning_content']
