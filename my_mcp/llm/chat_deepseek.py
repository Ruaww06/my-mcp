from dotenv import load_dotenv
import os

import json
from openai import AsyncOpenAI, APIError, RateLimitError
from utils import clean_reasoning_content

load_dotenv()
api_key = os.getenv("DS_API_KEY")
base_url = os.getenv("DS_BASE_URL")

class ChatDeepSeek:
    def __init__(self, api_key, base_url):
        self.model_name = "deepseek-reasoner"
        self.api_key = api_key
        self.base_url = base_url

    async def generate_with_deepseek_stream(self, client: AsyncOpenAI, conversation,
                                    formatted_functions):
        """Internal function for streaming generation"""
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                tools=[{"type": "function", "function": f} for f in formatted_functions],
                tool_choice="auto",
                stream=True
            )

            current_tool_calls = []
            current_content = ""

            async for chunk in response:
                delta = chunk.choices[0].delta
                
                if delta.content:
                    # Immediately yield each token without buffering
                    yield {"assistant_text": delta.content, "tool_calls": [], "is_chunk": True, "token": True}
                    current_content += delta.content

                # Handle tool call updates
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # Initialize or update tool call
                        while tool_call.index >= len(current_tool_calls):
                            current_tool_calls.append({
                                "id": "",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            })
                        
                        current_tool = current_tool_calls[tool_call.index]
                        
                        # Update tool call properties
                        if tool_call.id:
                            current_tool["id"] = tool_call.id
                        
                        if tool_call.function.name:
                            current_tool["function"]["name"] = (
                                current_tool["function"]["name"] + tool_call.function.name
                            )
                        
                        if tool_call.function.arguments:
                            # Properly accumulate JSON arguments
                            current_args = current_tool["function"]["arguments"]
                            new_args = tool_call.function.arguments
                            
                            # Handle special cases for JSON accumulation
                            if new_args.startswith("{") and not current_args:
                                current_tool["function"]["arguments"] = new_args
                            elif new_args.endswith("}") and current_args:
                                # If we're receiving the end of the JSON object
                                if not current_args.endswith("}"):
                                    current_tool["function"]["arguments"] = current_args + new_args
                            else:
                                # Middle part of JSON - append carefully
                                current_tool["function"]["arguments"] += new_args

                # If this is the last chunk, yield final state with complete tool calls
                if chunk.choices[0].finish_reason is not None:
                    # Clean up and validate tool calls
                    final_tool_calls = []
                    for tc in current_tool_calls:
                        if tc["id"] and tc["function"]["name"]:
                            try:
                                # Ensure arguments is valid JSON
                                args = tc["function"]["arguments"].strip()
                                if not args or args.isspace():
                                    args = "{}"
                                # Parse and validate JSON
                                parsed_args = json.loads(args)
                                tc["function"]["arguments"] = json.dumps(parsed_args)
                                final_tool_calls.append(tc)
                            except json.JSONDecodeError:
                                # If arguments are malformed, try to fix common issues
                                args = tc["function"]["arguments"].strip()
                                # Remove any trailing commas
                                args = args.rstrip(",")
                                # Ensure proper JSON structure
                                if not args.startswith("{"):
                                    args = "{" + args
                                if not args.endswith("}"):
                                    args = args + "}"
                                try:
                                    # Try parsing again after fixes
                                    parsed_args = json.loads(args)
                                    tc["function"]["arguments"] = json.dumps(parsed_args)
                                    final_tool_calls.append(tc)
                                except json.JSONDecodeError:
                                    # If still invalid, default to empty object
                                    tc["function"]["arguments"] = "{}"
                                    final_tool_calls.append(tc)

                    yield {
                        "assistant_text": current_content,
                        "tool_calls": final_tool_calls,
                        "is_chunk": False
                    }

        except Exception as e:
            yield {"assistant_text": f"OpenAI error: {str(e)}", "tool_calls": [], "is_chunk": False}

    async def generate_with_deepseek_sync(self, client: AsyncOpenAI, conversation, 
                                    formatted_functions):
        """Internal function for non-streaming generation"""
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                tools=[{"type": "function", "function": f} for f in formatted_functions],
                tool_choice="auto",
                stream=False
            )

            choice = response.choices[0]
            assistant_text = choice.message.content or ""
            tool_calls = []
            
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if tc.type == 'function':
                        tool_call = {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}"
                            }
                        }
                        # Ensure arguments is valid JSON
                        try:
                            json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            tool_call["function"]["arguments"] = "{}"
                        tool_calls.append(tool_call)
            return {"assistant_text": assistant_text, "tool_calls": tool_calls}

        except APIError as e:
            return {"assistant_text": f"OpenAI API error: {str(e)}", "tool_calls": []}
        except RateLimitError as e:
            return {"assistant_text": f"OpenAI rate limit: {str(e)}", "tool_calls": []}
        except Exception as e:
            return {"assistant_text": f"Unexpected OpenAI error: {str(e)}", "tool_calls": []}

    async def get_deepseek_response(self, conversation,
                                all_functions, stream = False):
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        formatted_functions = []
        for func in all_functions:
            formatted_func = {
                "name": func["name"],
                "description": func["description"],
                "parameters": func["parameters"]
            }
            formatted_functions.append(formatted_func)

        clean_reasoning_content(conversation)

        if stream:
            return self.generate_with_deepseek_stream(
                client, conversation, formatted_functions
            )
        else:
            return await self.generate_with_deepseek_sync(
                client, conversation, formatted_functions
            )
            

        