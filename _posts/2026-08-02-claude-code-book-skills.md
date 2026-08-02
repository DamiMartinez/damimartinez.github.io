---
layout: post
title: "book-skills: Turning Classic Books Into Claude Code Skills"
categories: [AI, Claude Code, Agent Skills, Open Source]
---

I just open-sourced `book-skills`: a Claude Code skill for each of my favorite books, so Claude applies that book's framework automatically the moment a conversation calls for it — no need to say the title out loud.

![book-skills banner](/images/book_skills_banner.png)

I've been listening to [Lenny's Podcast latest episode with Dianne Penn](https://www.youtube.com/watch?v=tivaWTTVRhY), Head of Product for Research & Labs at Anthropic, and one thing she said stuck with me: she'd built herself a set of Claude Code skills based on the management and product books that shaped how she works, so Claude applies that thinking automatically instead of her having to re-explain it every time.

That reminded me of Marc Andreessen's line about only reading books that are at least ten years old — the ones that have already proven durable. So I picked books I'd apply that filter to myself, and turned them into skills:

👉 [github.com/DamiMartinez/book-skills](https://github.com/DamiMartinez/book-skills)

It's public, MIT-licensed, and open to anyone who wants to add a skill for their own field.

## What this actually is

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) is a folder with a `SKILL.md` file: some frontmatter plus instructions that Claude loads only when it's relevant to what you're doing. `book-skills` is a collection of these, one per book — each one packaging that book's core framework so Claude applies it automatically whenever the *ideas* come up, not just when you say the title.

Concretely: with the `lean-startup` skill installed, saying "help me plan this new feature, I want a lean approach" or "how do I validate this before building it" makes Claude structure its answer around Build-Measure-Learn, MVPs, and actionable metrics — no need to say "use the lean startup skill."

## How triggering works

Claude Code decides whether to use a skill by matching what you're asking against that skill's `description` field. That's the whole mechanism — so a good skill description front-loads the book's actual terminology (validated learning, MVP, wartime CEO, pointer arithmetic) instead of just restating the title. You don't need to name the book; mentioning its concepts is enough. And skills only kick in for tasks substantial enough to benefit — a trivial one-off question won't necessarily trigger one, even if it's topically related.

## What's inside

Eight skills so far, spanning product, leadership, and engineering:

| Skill | Book | Topic |
|---|---|---|
| `lean-startup` | *The Lean Startup* — Eric Ries | MVP, Build-Measure-Learn, validated learning, pivot vs. persevere |
| `high-output-management` | *High Output Management* — Andrew S. Grove | Managerial leverage, task-relevant maturity, one-on-ones, decision-making |
| `hard-thing-about-hard-things` | *The Hard Thing About Hard Things* — Ben Horowitz | Wartime/peacetime CEO, no-easy-answers decisions, firing/layoffs |
| `pragmatic-programmer` | *The Pragmatic Programmer*, 20th Anniversary Ed. — Hunt & Thomas | DRY, orthogonality, coupling, Design by Contract, refactoring |
| `ai-modern-approach` | *Artificial Intelligence: A Modern Approach* — Russell & Norvig | PEAS, agent architectures, search, CSPs, Bayesian networks, MDPs |
| `machine-learning-mitchell` | *Machine Learning* — Tom M. Mitchell | Bias-variance, evaluation methodology, MAP/naive Bayes, PAC learning |
| `c-programming-language` | *The C Programming Language*, 2nd Ed. — Kernighan & Ritchie | Pointer arithmetic, declaration reading, manual memory ownership |
| `automate-boring-stuff` | *Automate the Boring Stuff with Python* — Al Sweigart | File ops, CSV/Excel/PDF handling, web scraping, safe automation |

More are coming — product management, design patterns, system architecture, data engineering — see the repo's contributing section.

## Anatomy of a skill

Every `SKILL.md` in the repo follows the same shape. Here's `lean-startup`, trimmed:

```markdown
---
name: lean-startup
description: Applies Eric Ries's "The Lean Startup" framework (Build-Measure-Learn
  loop, validated learning, MVP, innovation accounting, pivot vs. persevere,
  actionable vs. vanity metrics, five whys, small batches, split testing) to plan
  or evaluate a feature, product, or business idea. Use this skill whenever the
  user wants a "lean startup approach," mentions "validated learning," "MVP,"
  "minimum viable product," "build-measure-learn," "pivot," "innovation
  accounting," or asks how to test a risky assumption before building something
  fully — even if they don't say "lean startup" by name.
---

# The Lean Startup

## Core concepts

**Leap-of-faith assumptions** — Every new idea rests on assumptions that could
be wrong: the value hypothesis and the growth hypothesis. Identify these
explicitly before proposing a build plan.

**MVP (Minimum Viable Product)** — Not a smaller/crappier version of the final
product. It's the smallest experiment that generates validated learning about
a leap-of-faith assumption, with the least effort.

...

## How to apply this

1. State the leap-of-faith assumptions.
2. Rank assumptions by risk — test the riskiest one first.
3. Design the smallest MVP that produces evidence on it.
4. Define the actionable metric(s) — and what counts as failure.
5. Set a pivot/persevere checkpoint.
```

Three sections, always: frontmatter with a keyword-dense `description`, a **Core concepts** list (8-13 ideas that are actually actionable, not a book report), and a **How to apply this** procedure describing how Claude should structure its response when the skill fires.

## Installing it

Skills are just folders. Clone the repo, then symlink the ones you want:

```bash
git clone https://github.com/DamiMartinez/book-skills.git

# Global — available in every Claude Code session
ln -s "$(pwd)/book-skills/skills/lean-startup" ~/.claude/skills/lean-startup

# Or install all of them at once
for skill in book-skills/skills/*/; do
  name=$(basename "$skill")
  [ "$name" = "_template" ] && continue
  ln -s "$(pwd)/$skill" ~/.claude/skills/"$name"
done
```

Symlinking (rather than copying) means every skill stays current with a `git pull`. No restart needed — Claude Code picks them up at the start of a session.

## Contributing

The repo ships a `_template` folder for exactly this. Adding a skill for a book that shaped how you work:

1. `cp -r skills/_template skills/<book-slug>`
2. Fill in `SKILL.md` — a `description` listing the book's actual key terms (this is what drives triggering), 3-8 genuinely actionable **Core concepts**, and concrete **How to apply this** steps.
3. Test it in a real conversation and tune the description if it doesn't trigger.
4. Update the README table and open a PR.

Keep it focused on what's *actionable* from the book, not exhaustive coverage of it — that's the whole design philosophy.

## Why this matters

Dianne Penn built these for herself, privately, to speed up her own PM work. That's really the whole point of this repo — instead of every engineer, PM, or manager re-deriving the same lessons from *The Lean Startup* or *High Output Management* on their own, the framework just sits there as a skill, ready the moment the conversation needs it.

If you've got a book you keep mentally reapplying at work, I'd genuinely like a PR for it.

**Like this content?** Subscribe to my [newsletter](https://damianmartinezcarmona.substack.com/) to receive more tips and tutorials about AI, Data Engineering, and automation.
