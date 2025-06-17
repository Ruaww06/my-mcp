# my-mcp
## 说在前面
本项目学习于**dolphin-mcp**
由于**dolphin-mcp**十分成熟，我砍掉了许多个性化功能，只保留了我用到的  
仅支持DeepSeek/OpenAI 默认模型为"deepseek-reasoner"  
### 感谢dolphin-mcp的制作者们，我终于学会了mcp的相关知识

## 使用提醒
毋庸置疑，你需要提供`.env`文件，并在里面给出你的`DS_API_KEY`和`DS_BASE_URL`  
因为我主要是为DeepSeek制作的，并不打算修改使用的模型，若你想更改模型，请查看源码"llm"目录中的"chat_deepseek.py"文件  
使用时需要提供两个文件的地址，请在"process_mcp"目录里的"agent.py"文件中更改为你的文件路径  
1. **mcp_servers_config.json**  
2. **log_messages.log**


若你想使用流式传输/连续对话，请在"main.py"文件中，将`stream`的值改为`True`
