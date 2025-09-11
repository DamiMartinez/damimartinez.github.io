---
layout: post
title: Adding References & Context Sources to My AI Workflow
categories: [AI, Productivity, Coding]
---

This is a continuation (Part 2) of my previous post: [My New Development Workflow - I Don’t Write Code Anymore](https://damimartinez.github.io/ai-dev-workflow/).  

In this article I want to share how I manage **references and context sources** to help my coding agents (Cursor, Gemini CLI, etc.) stay accurate and up-to-date.


## References Folder in `ai_docs`

Inside my `ai_docs/` directory, I keep a subfolder called `references/`.  

- In `references/` I clone repos, documentation, and libraries that I may want to use as extra context.  
- I include instructions in `AGENTS.md` and in my task templates for when and how the agent should consult those references.  
- **Don’t forget** to add `references/` to `.gitignore`, because you don’t want to track or commit all that content in Git.

This setup helps when the coding task depends on specifics in a library or external doc. Having those references locally means the agent doesn’t have to guess or rely on possibly stale online info.


## Using Context7 MCP Server

Another source of context I rely on heavily is the **Context7 MCP server** ([upstash/context7](https://github.com/upstash/context7)).  

- It holds up-to­-date documentation for many code libraries.  
- In `AGENTS.md` and templates I include Context7 as a source of truth whenever tasks involve library usage, API changes, or newer features.  
- This helps ensure the agent has access to the freshest documentation, which reduces errors or outdated assumptions.


## Recycling Previous Implementations as References

One more thing I often do: I include an existing, well-structured repository as a reference, so I can reuse patterns, structure or boilerplate instead of starting fully from scratch.  

For example, I always include [my base-agent repo](https://github.com/DamiMartinez/base-agent) whenever I start a new ADK agent, to build on a basic scaffold of features.  

You could do the same with a frontend app boilerplate, API boilerplate, or whatever matches your project domain. The idea is: the agent sees real, working examples in your references so it can borrow useful parts rather than invent them.


## Why This Matters

Putting together live references + Context7 + the `AGENTS.md` + template system means fewer surprises:  

- My coding agents can check real, current docs when needed.  
- Updates to libraries are less likely to break things because the agent has reference material.  
- The workflow stays more stable, less “trial and error” when adding new dependencies or starting new projects.


If you’d like, I can share some examples of my `references/` folder, sample templates where I reference Context7, or even a sample `AGENTS.md` section showing these instructions. Just hit me up via LinkedIn or email.
