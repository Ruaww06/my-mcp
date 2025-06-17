from llm.chat_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

from typing import List, Dict
import json

import logging

from utils import load_config_from_file
from process_mcp.transport import StdioMCP
from process_mcp.transport import SSEMCP
from utils import clean_reasoning_content

logger = logging.getLogger('my_mcp')

load_dotenv()
api_key = os.getenv("DS_API_KEY")
base_url = os.getenv("DS_BASE_URL")

llm = ChatDeepSeek(api_key=api_key, base_url=base_url)

async def log_messages_to_file(messages: List[Dict], functions: List[Dict], log_path: str):
    """
    Log messages and function definitions to a JSONL file.

    Args:
        messages: List of messages to log
        functions: List of function definitions
        log_path: Path to the log file
    """
    try:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Append to file
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "messages": messages,
                "functions": functions
            }) + "\n")
    except Exception as e:
        logger.error(f"Error logging messages to {log_path}: {str(e)}")

async def process_tool_call(tc, servers: Dict[str, StdioMCP]):
    func_name = tc["function"]["name"]
    func_args_str = tc["function"].get("arguments", "{}")
    try:
        func_args = json.loads(func_args_str)
    except:
        func_args = {}
    
    parts = func_name.split("_", 1)
    if len(parts) != 2:
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": func_name,
            "content": json.dumps({"error": "Invalid function name format"})
        }
    srv_name, tool_name = parts
    print(f"\nView result from {tool_name} from {srv_name} {json.dumps(func_args)}")

    if srv_name not in servers:
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": func_name,
            "content": json.dumps({"error": f"Unknown server: {srv_name}"})
        }
    
    # Get the tool's schema
    tool_schema = None
    for tool in servers[srv_name].tools:
        if tool["name"] == tool_name:
            tool_schema = tool.get("inputSchema", {})
            break

    if tool_schema:
        # Ensure required parameters are present
        required_params = tool_schema.get("required", [])
        for param in required_params:
            if param not in func_args:
                return {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": func_name,
                    "content": json.dumps({"error": f"Missing required parameter: {param}"})
                }
            
    result = await servers[srv_name].call_tool(tool_name, func_args)
    print(json.dumps(result, indent=2))

    return {
        "role": "tool",
        "tool_call_id": tc["id"],
        "name": func_name,
        "content": json.dumps(result)
    }

# __init__ 是同步方法，通过@classmethod + create，可以使用异步初始化（如果需要在初始化中加载配置之类的
class MCPAgent:
    @classmethod
    async def create(cls, 
                     mcp_server_config_path,
                     log_messages_path,
                     stream = False):
        obj = cls()
        await obj._initialize(
            mcp_server_config_path=mcp_server_config_path,
            log_messages_path=log_messages_path,
            stream=stream
        )
        return obj
    
    def __init__(self):
        pass

    async def _initialize(self, 
                          mcp_server_config_path,
                          log_messages_path,
                          stream = False):
        self.stream = stream
        self.log_messages_path = log_messages_path

        # 加载 MCP服务器配置文件
        mcp_server_config = load_config_from_file(mcp_server_config_path)
        servers_cfg = mcp_server_config.get('mcpServers', {})

        # 启动 MCP服务器
        self.servers = {}
        self.all_functions = []
        for server_name, conf in servers_cfg.items():
            if "url" in conf:  # SSE server
                client = SSEMCP(server_name, conf["url"])
            else:  # Local process-based server
                client = StdioMCP(
                    server_name=server_name,
                    command=conf.get("command"),
                    args=conf.get("args", []),
                    env=conf.get("env", {}),
                    cwd=conf.get("cwd", None)
                )

            ok = await client.start()
            if not ok:
                print(f"[WARN] Could not start server {server_name}")
                continue
            else:
                print(f"[OK] {server_name}")
            
            # 整理可用工具
            tools = await client.list_tools()
            for t in tools:
                input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                fn_def = {
                    "name": f"{server_name}_{t['name']}",
                    "description": t.get("description", ""),
                    "parameters": input_schema
                }
                self.all_functions.append(fn_def)

            self.servers[server_name] = client
        
        if not self.servers:
            error_msg = "No MCP servers could be started."
            return error_msg
        
        self.conversation = []

        # 建立对话
        system_msg = "You are a helpful assistant."
        self.conversation.append({"role": "system", "content": system_msg})

    async def cleanup(self):
        """Clean up servers and log messages"""
        if self.log_messages_path:
            await log_messages_to_file(self.conversation, self.all_functions, self.log_messages_path)
        for cli in self.servers.values():
            await cli.stop()
        self.servers.clear()

    async def prompt(self, user_query):
        self.conversation.append({"role": "user", "content": user_query})
        if self.stream:
            async def stream_response():
                try:
                    while True:  # Main conversation loop
                        generator = await llm.get_deepseek_response(self.conversation, self.all_functions, stream=True)
                        accumulated_text = ""
                        tool_calls_processed = False
                        
                        async for chunk in generator:
                            if chunk.get("is_chunk", False):
                                # Immediately yield each token without accumulation
                                if chunk.get("token", False):
                                    yield chunk["assistant_text"]
                                accumulated_text += chunk["assistant_text"]
                            else:
                                # This is the final chunk with tool calls
                                if accumulated_text != chunk["assistant_text"]:
                                    # If there's any remaining text, yield it
                                    remaining = chunk["assistant_text"][len(accumulated_text):]
                                    if remaining:
                                        yield remaining
                                
                                # Process any tool calls from the final chunk
                                tool_calls = chunk.get("tool_calls", [])
                                if tool_calls:
                                    # Add type field to each tool call
                                    for tc in tool_calls:
                                        tc["type"] = "function"
                                    # Add the assistant's message with tool calls
                                    assistant_message = {
                                        "role": "assistant",
                                        "content": chunk["assistant_text"],
                                        "tool_calls": tool_calls
                                    }
                                    self.conversation.append(assistant_message)
                                    
                                    # Process each tool call
                                    for tc in tool_calls:
                                        if tc.get("function", {}).get("name"):
                                            result = await process_tool_call(tc, self.servers)
                                            if result:
                                                self.conversation.append(result)
                                                tool_calls_processed = True
                        
                        # Break the loop if no tool calls were processed
                        if not tool_calls_processed:
                            break
                        
                finally:
                    pass
            return stream_response()
        else:
            try:
                final_text = ""
                while True:
                    gen_result = await llm.get_deepseek_response(self.conversation, all_functions=self.all_functions)
                    assistant_text = gen_result['assistant_text']
                    final_text = assistant_text
                    tool_calls = gen_result.get('tool_calls', [])

                    # 清除 reasoning_content
                    clean_reasoning_content(self.conversation)

                    assistant_msg = {"role": "assistant", "content": assistant_text}
                    if tool_calls:
                        for tc in tool_calls:
                            tc["type"] = "function"
                        assistant_msg["tool_calls"] = tool_calls
                    self.conversation.append(assistant_msg)
                    logger.info(f"Added assistant message: {json.dumps(assistant_msg, indent=2)}")

                    if not tool_calls:
                        break

                    for tc in tool_calls:
                        result = await process_tool_call(tc, self.servers)
                        if result:
                                self.conversation.append(result)
                                logger.info(f"Added tool result: {json.dumps(result, indent=2)}")
                
            finally:
                return final_text

async def run_interaction(user_query, mcp_config_path, log_messages_path, stream=False):
    agent = await MCPAgent.create(
        mcp_server_config_path=mcp_config_path,
        log_messages_path=log_messages_path,
        stream=stream
    )
    response = await agent.prompt(user_query=user_query)
    await agent.cleanup()
    return response

