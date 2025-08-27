---
layout: post
title: AI Tools and Resources I Use in My Day-to-Day Work
categories: [AI, Tools, Productivity]
---

Over the past months, I’ve been working a lot on developing AI agents and experimenting with different tools. Here’s an overview of the main AI tools and resources I currently use on a daily basis.

---

## 1. My IDE: Cursor

The main editor I use is **[Cursor](https://cursor.com/home)**. It’s very easy to use because it’s a fork of **VS Code**, so if you’ve worked with VS Code before, switching to Cursor is simple.

One of the greatest advantages of Cursor is the possibility to connect **MCP servers**. For example, the [context7 MCP](https://github.com/upstash/context7) gives you access to the **most up-to-date documentation of Python and other libraries**. This means that even if the model itself doesn’t know about the latest updates, Cursor will always check the most recent docs through context7.

Another powerful feature is the ability to define **templates and rules**. With these, you can enforce Cursor to always follow your coding requirements and best practices. Over time, you can keep refining these templates with more conditions so that the model becomes better aligned with your personal development style.

👉 Highly recommended video to see Cursor’s full potential:  
[Maximize your productivity with Cursor](https://www.youtube.com/watch?v=nO9ly_ZDiUE)

By default, **Cursor Pro** integrates:
- **Anthropic’s Sonnet models**
- **OpenAI’s GPT models**
- **xAI’s Grok**
- (and you can add more models if you want)

💡 **Pro tip for Cursor:**  
To optimize memory and reduce token usage, don’t keep everything in a single chat. Instead, open **multiple chats, one per task**. This avoids long conversations with too much context, which increases cost and slows down responses.

💲 **Pricing:** Cursor Pro costs **$20/month**, and in normal daily development you rarely hit the usage limits.

---

## 2. Gemini CLI

I also use **Gemini CLI** when coding. Sometimes I use it to double-check or get a second opinion on the code produced in Cursor.

A great thing about Gemini CLI is that it gives access to the **latest Gemini models** (currently Gemini 2.5 Pro).

💲 **Pricing tip:** For daily development, Gemini CLI is almost always free. Normally you don’t reach the usage limit of the top-tier **gemini-2.5-pro** model, and if you ever do, you can just switch to the **gemini-2.0** model.

---

## 3. Terminal Setup  

For package management and running my projects I use:  
- **Poetry**  
- **uv**

---

## 4. Running Models Locally  

When I want to run models locally, I use **[Ollama](https://ollama.com/)**. It makes downloading and testing models on your own machine easy.

---

## 5. Building and Deploying AI Agents  

As I mentioned in my [previous article](https://damimartinez.github.io/adk-issues-pt1/), I mainly use **Google ADK** for developing agents.

For deployment, I rely on **Vertex AI Agent Builder**:
- It’s framework- and model-agnostic.
- Easy and quick to use.
- Very cost-effective compared to other platforms.

---

## 6. Research & Exploration  

For research I use:  
- **Claude** (for deeper reasoning/explanations)  
- **Gemini** (via **Gemini CLI** or **Google AI Studio**)

Google AI Studio also supports **RAG directly**.

Another great tool is **[NotebookLM](https://notebooklm.google.com/)**:
- You can upload your own documents or notes.
- It builds an interactive research assistant powered by Gemini.
- Perfect for exploring technical papers or custom datasets.

---

## 7. AI for Media & Content Generation  

For writing articles like this one or polishing drafts, I normally use **ChatGPT**. It helps structure ideas, improve clarity, and save time.

For creative tasks, I also experiment with:  
- **Sora** (via ChatGPT) → text-to-video  
- **Imagen** (Google) → images  
- **Veo 3** (Google) → videos

---

## 8. Useful Resources  

Here’s a list of resources I keep going back to:

- 📘 [Google ADK Documentation](https://google.github.io/adk-docs/)  
- 💻 [Google ADK Samples](https://github.com/google/adk-samples)  
- 📚 [Google Gemini Cookbook (ML + examples)](https://github.com/google-gemini/cookbook)  
- 🌐 [Model Context Protocol docs](https://modelcontextprotocol.io/docs/getting-started/intro)  
- 🛒 [MCP Marketplace](https://mcp.so/)  
- ▶️ [Google Developers YouTube](https://www.youtube.com/@GoogleDevelopers)  
- ▶️ [Anthropic Streams](https://www.youtube.com/@anthropic-ai/streams)  
- ▶️ [AI with Brandon (YouTube channel)](https://www.youtube.com/@aiwithbrandon)

---

## Final Thoughts  

Developing AI agents and experimenting with models requires a mix of tools — from IDEs like Cursor to deployment platforms like Vertex AI.

For me, **Cursor and Gemini CLI are the backbone of my daily workflow**. They complement each other well: Cursor is great for structured coding with rules and MCPs, while Gemini CLI is a reliable, free companion to validate or contrast results.

This is just my current setup — tools evolve fast, and I’ll keep sharing updates (and model comparisons) in future posts.
