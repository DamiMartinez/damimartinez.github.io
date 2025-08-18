---
layout: post
title: Lessons Learned from Using Google ADK - Part 1
categories: [AI, Google ADK, Agent Development]
---

I have been working with the **Google Agent Development Kit (ADK)** for the past couple of months. While it is a powerful framework, the documentation is limited and I had to solve several unexpected problems on my own.  

Here is a summary of the main issues I faced and how I solved them.  

⚠️ **Note:** all of these observations are based on my experience with  
`google-cloud-aiplatform[adk,agent-engines]==1.100.0`.  
Since this is very new technology, the SDK is evolving quickly and the Google team is constantly fixing and improving it.  



## 1. `errlog=None` is required for MCP tools  
If you use MCP tools, you must set `errlog=None` in the `MCPToolset`. Without it, the agent will not deploy correctly on VertexAI.  

```python
tools = [
    MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="uvx",
                args=["mcp-server-fetch"],
            ),
            timeout=10,
        ),
        errlog=None  # required for deployment on VertexAI
    )
]
```



## 2. Remember `load_dotenv()`  
If you forget to load environment variables with `dotenv`, your agent will deploy, but it will fail at runtime with vague error messages.  

```python
from dotenv import load_dotenv
from vertexai.preview.agents import Agent

# Load environment variables first
load_dotenv()

# continue with your agent setup...
```



## 3. The `thought_signature` error  
Sometimes the agent suddenly stops responding and only returns strange hashes instead of messages:  

```json
{
  "content": {
    "parts": [
      {
        "thought_signature": "CpUCAcu98PDiQfIyH8Qj9nB2UX_OMAAoF..."
      }
    ]
  }
}
```  

This usually happens because of an update in the **VertexAI SDK** or the **model**.  
The solution is to either:  
- Update the `vertexai` SDK to the latest version, or  
- Downgrade the model version in your configuration.  



## 4. RAG does not work with sub-agents  
If you want to use **RAG**, you need to attach it with `agenttool` to the root agent.  
It does not work reliably with sub-agents, so always keep RAG on the main agent definition.  



## 5. User and session IDs  
You can access `user_id` and `session_id` from the **callback context**.  

⚠️ Be aware that sub-agents run with their own IDs, which are different from the root agent.  

👉 **Recommendation:** I save the IDs in the root agent state as soon as possible, using the `before_agent_callback`. This way you don’t need to worry about them later.  

Example:  

```python
def before_agent_callback(callback_context: CallbackContext) -> None:
    user_id = callback_context._invocation_context.user_id
    session_id = callback_context._invocation_context.session.id
    
    # Save in state for later use
    callback_context.state['user_id'] = user_id
    callback_context.state['session_id'] = session_id
```



## 6. Use region `us-central1`  
When deploying your agent, I recommend using **`us-central1`**. This region receives new features first.  

If you deploy in another region, you may find that some features are missing or that the deployment fails without clear explanations.  



## General Lessons
The main lesson I learned is that **deploying an AI agent is not easy**. Even with a framework like ADK, the first deployment will take time and effort.  

That being said, **Vertex AI Agent Engine** simplifies the process a lot. It removes much of the complexity of setting up infrastructure and lets you focus on the agent logic itself.  

To make it easier for others, I created a simple **base agent** that you can use as a starting point:  
👉 [github.com/DamiMartinez/base-agent](https://github.com/DamiMartinez/base-agent)  

It includes a deployment script and basic setup. I plan to extend it with more features such as MCP tools, callbacks, evaluations, and more — if people find it useful.  



## Final note  
The ADK has great potential, but many of these details are undocumented.  
I hope these notes help others avoid the same issues I encountered.  

Again, these are based on my experience with version `1.100.0` of the SDK, and things may change as Google continues to improve the platform.  
