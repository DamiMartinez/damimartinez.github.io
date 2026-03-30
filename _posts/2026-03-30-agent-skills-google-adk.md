---
layout: post
title: "Agent Skills in Google ADK: Build a Scraping Agent with Progressive Disclosure"
categories: [AI, Google ADK, Agent Skills, Web Scraping]
---

A few weeks ago I wrote about [Agent Skills in Gemini CLI](https://damimartinez.github.io/agent-skills-gemini-cli/) and how they solve the context bloat problem by loading specialized instructions on demand. The same open standard is now available in **Google ADK (Agent Development Kit)** for Python — and it opens up a whole new design space for building agents in code.

In this post I want to show you exactly how it works, using a real project as the guide: **[ScrapeAgent](https://github.com/DamiMartinez/scrapeagent)**, an open-source web scraping agent I built on top of ADK skills.

## Quick recap: what are Agent Skills?

If you haven't read the previous post, here is the short version.

An **Agent Skill** is a self-contained unit of functionality — instructions, reference documents, and assets — packaged in a folder with a `SKILL.md` file at its root. The clever part is how they load:

- **L1 (Metadata):** Only the skill's `name` and `description` are loaded at startup. This costs ~100 tokens per skill, no matter how rich the skill actually is.
- **L2 (Instructions):** The full body of `SKILL.md` is loaded only when the agent decides this skill is relevant to the current task.
- **L3 (Resources):** Files in `references/` and `assets/` subdirectories are loaded only if the skill's instructions reference them.

This **progressive disclosure** model means you can keep a library of dozens of skills without blowing up your context window on every single request.

Skills for ADK are available in **google-adk v1.25.0+** (currently experimental) and follow the [Agent Skills open specification](https://agentskills.io/specification) — the same standard used by Gemini CLI, Cursor, and others.

## How skills work in ADK

The entry point is the `SkillToolset` class. You load one or more `Skill` objects and pass the toolset to your agent's `tools` list:

```python
from google.adk import Agent
from google.adk.tools import skill_toolset

my_skill_toolset = skill_toolset.SkillToolset(skills=[weather_skill, news_skill])

root_agent = Agent(
    model="gemini-2.5-flash",
    name="my_agent",
    instruction="You are a helpful assistant.",
    tools=[my_skill_toolset],
)
```

At startup the agent only sees the names and descriptions of each skill. When a user request matches a skill's description, the agent loads the full instructions and goes to work. No manual wiring, no giant system prompt.

### Defining a skill: the directory structure

The recommended layout is:

```
my_agent/
    agent.py
    skills/
        my-skill/
            SKILL.md          # required
            references/
                REFERENCE.md  # detailed docs, workflows, guides
            assets/
                template.csv  # data files, schemas, examples
```

Only `SKILL.md` is required. The `references/` and `assets/` directories are optional — but they are what makes a skill genuinely powerful, because they let you offload rich documentation to L3 without paying for it until the skill is actually activated.

### Anatomy of a SKILL.md

```markdown
---
name: my-skill
description: What this skill does and when to use it. Mention specific triggers.
metadata:
  author: yourname
  version: "1.0"
---

## Instructions

Step-by-step instructions that tell the agent exactly what to do when this skill is active.
Reference any files in references/ or assets/ by name — ADK will load them on demand.
```

The YAML frontmatter is the L1 layer — it is always in context. Everything below the second `---` is L2 and loaded only on activation.

## ScrapeAgent: a real ADK skills project

Let me walk through how I put all of this together in [ScrapeAgent](https://github.com/DamiMartinez/scrapeagent).

The idea is simple: instead of hardcoding scraping logic in Python, the agent's expertise lives in `SKILL.md` files. Each skill describes how to scrape a specific website — which CSS selectors to use, whether to use a static fetcher or a full browser, how to handle pagination, and so on. No Python required to add a new scraper.

The project structure looks like this:

```
scrapeagent/
    agent.py            # root agent definition
    prompt.py           # system prompt
    skills/
        hacker-news/
            SKILL.md    # static HTML scraper
        github-trending/
            SKILL.md    # JS-rendered scraper
        skill-creator/
            SKILL.md    # meta-skill: creates new skills at runtime
            assets/
                SKILL_TEMPLATE.md
    tools/
        file_tools.py   # save_output() and create_skill()
```

### Loading skills at startup

As of **google-adk v1.25.0**, `load_skill_from_dir` is built into the package — no custom loader needed. It reads the `SKILL.md` frontmatter and body, and automatically discovers any `references/` and `assets/` files:

```python
from google.adk.skills import load_skill_from_dir
```

That single import replaces the custom `skills_loader.py` that earlier versions of ScrapeAgent shipped.

### The root agent

`agent.py` scans the `skills/` directory, loads every skill, wraps them in a `SkillToolset`, and wires everything together with two MCP toolsets (a lightweight HTTP fetcher and a Playwright browser) plus two custom tools:

```python
from google.adk.skills import load_skill_from_dir

_skills_dir = pathlib.Path(__file__).parent / "skills"
_skills = [
    load_skill_from_dir(d) for d in sorted(_skills_dir.iterdir()) if d.is_dir()
]

skill_tools = skill_toolset.SkillToolset(skills=_skills)

# mcp-server-fetch: static HTML (no JS rendering)
fetch_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command="uvx", args=["mcp-server-fetch"]),
        timeout=30.0,
    )
)

# @playwright/mcp: full browser for JS-heavy sites
playwright_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx", args=["-y", "@playwright/mcp", "--headless"]
        ),
        timeout=30.0,
    )
)

root_agent = LlmAgent(
    model=LiteLlm(model=os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash")),
    name="scrapeagent",
    instruction=PROMPT,
    tools=[skill_tools, fetch_toolset, playwright_toolset, save_output, create_skill],
)
```

One thing worth noting: the `timeout=30.0` on both MCP toolsets is intentional. The default of 5 seconds is too short when launching `npx` or `uvx` for the first time — they need to download packages. Bump the timeout or you will see random failures on cold starts.

### What a skill looks like in practice

Here is the `hacker-news` skill in full — it is the simplest example in the repo and a good template for writing your own:

```markdown
---
name: hacker-news
description: Scrape Hacker News front page stories including titles, URLs, scores,
  authors, and comment counts from news.ycombinator.com. Use when the user mentions
  Hacker News, HN, ycombinator, or wants tech news stories.
metadata:
  author: scrapeagent
  version: "1.0"
---

# Hacker News Scraper

## Overview
Scrapes the Hacker News front page at https://news.ycombinator.com
No JavaScript required — use the `fetch` tool.

## Instructions

1. Fetch `https://news.ycombinator.com` using the fetch tool
2. The page has 30 story items. Each story spans two `<tr>` elements:
   - Row 1 (`.athing`): title and link
   - Row 2 (`.subtext`): metadata (score, author, comments)

3. For each story extract:
   - `rank`: `.rank` text, strip the trailing `.`
   - `title`: `.titleline > a` text
   - `url`: `.titleline > a` href
   - `score`: `.score` text, extract the number
   - `author`: `.hnuser` text
   - `age`: `.age a` text
   - `comments`: last `<a>` in `.subtext` — extract number, or 0 if "discuss"

4. Save with columns: rank, title, url, domain, score, author, age, comments
```

When you ask the agent "scrape Hacker News and save as CSV", it recognizes the match, loads this skill, and follows the instructions exactly — no ambiguity, no hallucinated selectors. The scraping knowledge is explicit, versioned, and reviewable.

## Defining skills in code

File-based SKILL.md is not the only option. ADK also lets you define skills directly in Python using the `models.Skill` class:

```python
from google.adk.skills import models

hacker_news_skill = models.Skill(
    frontmatter=models.Frontmatter(
        name="hacker-news",
        description=(
            "Scrape Hacker News front page stories including titles, URLs, scores, "
            "authors, and comment counts. Use when the user mentions Hacker News, "
            "HN, ycombinator, or wants tech news stories."
        ),
    ),
    instructions=(
        "Fetch https://news.ycombinator.com using the fetch tool. "
        "Each story spans two <tr> elements: .athing (title/link) and .subtext (metadata). "
        "Extract rank, title, url, score, author, age, comments for each of the 30 stories."
    ),
    resources=models.Resources(
        references={
            "selectors.md": (
                "- `.athing`: story row with rank and title\n"
                "- `.subtext`: metadata row with score, author, age, comments\n"
                "- `.titleline > a`: story link element\n"
                "- `.score`, `.hnuser`, `.age a`: individual metadata fields"
            )
        }
    ),
)

my_skill_toolset = skill_toolset.SkillToolset(skills=[hacker_news_skill])
```

The three levels still apply: `frontmatter` is L1, `instructions` is L2, and anything in `resources` is L3. The difference is that the content lives in your Python source rather than on disk.

**When to choose inline skills over file-based:**

- **Inline** works well when skills are generated dynamically — built from a database query, fetched from an API, or assembled from user configuration at startup.
- **File-based** is better when skills are human-editable recipes you want tracked in version control. Anyone who can write Markdown can add or update a skill without touching Python.

ScrapeAgent uses the file-based approach by design: the goal is to let a data analyst contribute a new scraping recipe with nothing but a `SKILL.md` file. But if your use case calls for runtime-generated skills, the `models.Skill` API gives you full programmatic control.

## Adding a new skill

There are two ways to extend the agent with a new website:

**Option 1 — Ask the agent.** The `skill-creator` skill is a meta-skill that writes new `SKILL.md` files for you. Just describe what you want to scrape:

```text
Create a skill to scrape the top posts from reddit.com/r/python
```

The agent activates `skill-creator`, explores the site structure, and writes a new skill directory. Restart the agent and the new skill is available automatically.

**Option 2 — Write it yourself.** Create a new directory under `scrapeagent/skills/` and drop in a `SKILL.md` following the structure above. No Python required — the skill loader picks it up on the next startup.

Either way, the workflow is the same: the expertise lives in Markdown, the Python code stays minimal.

## Running ScrapeAgent

```bash
git clone https://github.com/DamiMartinez/scrapeagent
cd scrapeagent
cp .env.example .env  # add your API key
poetry install
adk web
```

Open the ADK web UI at `http://localhost:8000`, pick the `scrapeagent` agent, and start asking it to scrape things:

- "Scrape the front page of Hacker News and save as CSV"
- "Get today's trending GitHub repos"
- "Create a skill to scrape quotes from quotes.toscrape.com"

## Why this design matters

The combination of Agent Skills and ADK produces something genuinely useful: an agent where **adding capability requires no Python**. A data analyst who can write Markdown can contribute a new scraping recipe. A team can maintain a library of 50 site-specific skills without any of them showing up in the context window until they are needed.

This is the same principle I described in the Gemini CLI post — progressive disclosure keeps your context lean and your agent focused — but now it is available programmatically, in code, as part of your application architecture.

If you are building agents with ADK, I would encourage you to think about what knowledge in your system could be moved into skills. API documentation, domain-specific workflows, site-specific extraction logic — anything that is rich, specialized, and only relevant to a subset of requests is a candidate.

**Like this content?** Subscribe to my [newsletter](https://damianmartinezcarmona.substack.com/) to receive more tips and tutorials about AI, Data Engineering, and automation.
