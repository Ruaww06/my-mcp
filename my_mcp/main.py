from process_mcp.agent import run_interaction
from process_mcp.agent import MCPAgent

import asyncio

import logging
import sys


mcp_config_path = "<你的mcp服务器配置文件地址>"
log_path = "<你的日志文件地址>"

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set logging level to DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # Log to stderr
    )
    logger = logging.getLogger("dolphin_mcp") # Get logger instance after basicConfig
    logger.debug("Logging configured at DEBUG level.")

    user_query = None
    stream = False # default 
    if stream:
        logger.debug("Interactive mode enabled.")
        agent = await MCPAgent.create(
            mcp_server_config_path=mcp_config_path,
            log_messages_path=log_path,
            stream=True # Interactive mode implies streaming
        )
        loop = asyncio.get_event_loop()
        try:
            while True:
                current_query = ""
                if user_query: # Use initial query first if provided
                    current_query = user_query
                    print(f"> {current_query}") # Simulate user typing the initial query
                    user_query = None # Clear after use
                else:
                    try:
                        user_input = await loop.run_in_executor(None, input, "> ")
                    except EOFError: # Handle Ctrl+D
                        print("\nExiting interactive mode.")
                        break 
                    if user_input.lower() in ["exit", "quit"]:
                        print("Exiting interactive mode.")
                        break
                    if not user_input:
                        continue
                    current_query = user_input
                
                if not current_query.strip(): # If after all that, query is empty, continue
                    continue

                print("AI: ", end="", flush=True)

                response_generator = await agent.prompt(current_query)
                full_response = ""
                async for chunk in response_generator:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print() # Add a newline after the full response
                
                # In a real chat, we might add full_response to a history
                # For now, each input is a new prompt in the same session

        finally:
            await agent.cleanup()
            logger.debug("Agent cleaned up.")
    else:
        user_query = input("请输入你的问题：")
        response = await run_interaction(user_query=user_query,
                            mcp_config_path=mcp_config_path,
                            log_messages_path=log_path)

        print(r"\n" + response.strip() + "\n")

if __name__ == "__main__":
    asyncio.run(main()) # Changed to asyncio.run(main())
