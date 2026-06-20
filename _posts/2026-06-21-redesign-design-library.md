---
layout: post
title: "How I Redesigned My Website Using an AI-Curated Design Library"
categories: [AI, Design, Productivity]
---

If you've visited this site recently, you may have noticed it looks different. Bolder. Blockier. More attitude. That's the neo-brutalist redesign — and the way I got there is actually a story about a workflow tip I picked up from a friend, one that I think is genuinely useful for anyone who builds websites.

## The conversation that started it

A few months ago I was talking with my friend **[Dimitar Inchev](https://www.linkedin.com/in/inchev/)** about AI tools and workflows. At some point he mentioned a habit he'd developed: every time he came across a website whose design he liked, he would document its entire visual system in a structured markdown file. Colors, typography, spacing, components, interactions, tone — all of it captured while the details were still fresh.

His reasoning was simple. Screenshots are useful but passive. A structured document is something you can *use* — give it to an AI agent and ask it to replicate or adapt the style. The collection grows over time and becomes a personal library of designs you've curated yourself, from sites you actually like.

I thought this was a smart idea, so I built a template based on what he described and put it to work straight away.

## The template

The template I created covers every meaningful design dimension of a website:

- **Colors** — every role in the palette (background, surface, primary text, accent, border) with exact hex values and a note on the color philosophy
- **Typography** — font family, weight, size, and line height for every text role, from display headings down to captions and code
- **Spacing & layout** — grid system, max content width, section padding, border radius, base unit
- **Visual style** — how shadows, borders, images, icons, and textures are handled
- **Components** — how the nav, hero, buttons, cards, forms, and footer actually behave
- **Interactions & animation** — hover states, transitions, scroll effects, loading states
- **Tone & personality** — a short paragraph describing the emotional feel of the design

Here's the full template:

```markdown
# [Design Name]

## Meta
- **Source:** [URL or description of where this design is from]
- **Style keywords:** [e.g. minimal, dark, editorial, brutalist, glassmorphism, corporate, playful]
- **Best for:** [e.g. SaaS landing page, portfolio, e-commerce, dashboard, blog]

---

## Colors

| Role | Name | Hex | Usage |
|------|------|-----|-------|
| Primary | | `#000000` | CTAs, key highlights |
| Secondary | | `#000000` | Supporting elements |
| Background | | `#000000` | Page background |
| Surface | | `#000000` | Cards, panels |
| Text primary | | `#000000` | Body copy, headings |
| Text secondary | | `#000000` | Captions, metadata |
| Accent | | `#000000` | Hover states, decorative |
| Border | | `#000000` | Dividers, outlines |

---

## Typography

| Role | Font family | Weight | Size (desktop) | Size (mobile) | Transform |
|------|-------------|--------|----------------|---------------|-----------|
| Display / H1 | | | | | |
| H2 | | | | | |
| H3 | | | | | |
| Body | | | | | |
| Caption / small | | | | | |
| Label / tag | | | | | |
| Monospace / code | | | | | |

**Font source:** [Google Fonts / Adobe / self-hosted / system]
**Line height:** body `1.x`, headings `1.x`
**Letter spacing:** headings `0.0xem`, labels `0.0xem`

---

## Spacing & Layout

- **Grid:** [e.g. 12-col, 8-col, free-flow]
- **Max content width:** [e.g. 1280px]
- **Section padding (vertical):** [e.g. 80px desktop / 48px mobile]
- **Component gap:** [e.g. 24px]
- **Base unit:** [e.g. 8px]
- **Border radius:** [e.g. none / 4px / 8px / full pill]

---

## Visual Style

- **Shadows:** [e.g. none / subtle `0 1px 3px rgba(0,0,0,0.1)` / dramatic]
- **Borders:** [e.g. hairline 1px / none / colored]
- **Images:** [e.g. full-bleed, rounded, grayscale, with overlay]
- **Icons:** [e.g. outline / filled / custom SVG / Lucide / Heroicons]
- **Texture / background:** [e.g. flat color, gradient, noise, grid, dots]

---

## Components

### Navigation
[Describe: sticky/fixed/static, logo position, link style, CTA button style, mobile behavior (hamburger, drawer, etc.)]

### Hero / Above the fold
[Describe: layout (centered / split / full-bleed image), headline size, subheadline, CTA layout, background treatment]

### Buttons
- Primary: [e.g. filled, pill shape, uppercase label, hover: darken 10%]
- Secondary: [e.g. outline, square corners, hover: fill]
- Ghost / text: [e.g. underline on hover]

### Cards
[Describe: background, border, radius, shadow, hover behavior, image treatment]

### Forms / Inputs
[Describe: border style, focus ring, label position (floating / above), error state]

### Footer
[Describe: columns, background, link style, legal text size]

---

## Interactions & Animation

- **Default transition:** [e.g. `all 150ms ease`]
- **Hover effects:** [e.g. lift with shadow, color shift, underline slide-in]
- **Scroll animations:** [e.g. none / fade-up on enter / parallax]
- **Page transitions:** [e.g. none / fade / slide]
- **Loading state:** [e.g. skeleton screens / spinner / shimmer]

---

## Tone & Personality

[2–4 sentences describing the overall feel: Is it cold and technical? Warm and approachable? Dense with information? Breathable and sparse? What emotion does it evoke?]

---

## Notes & Reuse Tips

[Anything that's hard to capture structurally — quirks, what makes this design distinctive, what to steal vs. what to skip, specific implementation gotchas.]
```

That last section might be the most underrated. When you've captured what makes a design *feel* the way it does, you can make judgment calls later without going back to the original site.

## Building the library

The workflow for adding a new entry is simple: find a website whose design you like, give the template to your AI coding agent, and prompt it to inspect the site and fill in every section. The agent reads the live page, extracts the colors, fonts, spacing, component patterns, and interaction details, and returns a completed spec file. No manual digging through DevTools required.

I started going through websites I'd bookmarked over the years and running them through this process. Not all of them — only the ones where I thought "I'd actually want to build something that looks like this."

Each entry taught me something different about how design decisions compound. Hard shadows are meaningless without chunky borders to match. A distinctive background color works because of its warmth, not just its value. A swappable accent system is clever but only worth the complexity if the personality of the site supports it.

Over time the library became something genuinely useful — a set of reference specs I could query rather than a folder of screenshots I'd never open again.

## The redesign

When I decided it was time to update this site, I knew I wanted to go neo-brutalist. Instead of describing a vibe and hoping the AI agent would guess right, I gave it something concrete: all the neo-brutalist templates in my library.

The prompt was roughly: *look at all the neo-brutalist style definitions in these files and use them as reference to redesign the site — extract the patterns that appear across multiple sources and use them as the foundation*.

The agent read through the templates, identified what the style actually requires structurally (hard shadows, visible borders, high-contrast chunky type, no gradients, no blur), and applied those principles to the site's existing layout. It wasn't blending the designs randomly — it was synthesizing what made them all *feel* the same despite looking different in their details.

## Cherry-picking specifics

After the initial pass, I went back through my templates and picked specific elements I wanted to bring in from particular sources:

- The **shadow system** and **border thickness** — exact values pulled straight from one of the templates
- The **button style** — pill shape, thick border, no gradient, fills on hover
- The **color approach** — near-black ink color for both text and shadows, keeping the palette tight

This is where having the exact values documented matters. "Use a hard shadow" is ambiguous. `3px 3px 0 var(--color-ink)` is not.

## Why this workflow is worth it

The design library strategy solves a real problem with AI-assisted design work: *describing what you want is hard*. Words like "clean", "modern", or "minimal" mean nothing specific. But a markdown file that says the body font is Atkinson Hyperlegible at 17px with 1.5 line height, and every card has a 2px solid border and a flat 5px hard shadow — that's a brief an agent can actually execute.

Over time the library compounds. You're not just collecting screenshots; you're collecting *knowledge* about why specific sites work. Each new template you write deepens your own understanding of design, not just your library.

If you spend any time building or updating websites, I'd recommend starting one. The template is simple to fill in while looking at a site you like, and even a library of five or ten entries gives you real leverage when you want to build something new.
