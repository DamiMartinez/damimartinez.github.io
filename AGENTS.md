# AGENTS.md

This file provides guidance for coding agents working in this repository.

## What this is

Personal blog/website for Damian Martinez Carmona, built with Jekyll using the [Reverie](https://github.com/amitmerchant1990/reverie) theme. Hosted on GitHub Pages at `https://damimartinez.github.io`.

## Commands

**Local development** (requires Ruby 2.4.3 and Bundler):
```sh
bundle install
bundle exec jekyll serve
```

The site is served at `http://localhost:4000`. Changes to `_config.yml` require a server restart; all other changes rebuild automatically.

**Build only:**
```sh
bundle exec jekyll build
```

Output goes to `_site/` (not committed).

## Architecture

- **`_config.yml`** — Site-wide settings: title, author, social links, Google Analytics (GA4), plugins, pagination (6 posts/page), permalink style (`/:title/`).
- **`_posts/`** — Blog posts in `YYYY-MM-DD-title.md` format with YAML front matter (`layout`, `title`, `categories`).
- **`_pages/`** — Static pages (`archive.md`, `categories.md`, `search.md`, `getting-started.md`). Included via `include: ['_pages']` in `_config.yml`.
- **`_layouts/`** — Only `page.html` is customized; the main `default` layout comes from the Reverie gem.
- **`_includes/`** — Overrides for analytics, Disqus comments, and SEO meta tags.
- **`_sass/`** — Sass partials for syntax highlighting (`_darcula.scss`, `_highlights.scss`) and CSS reset.
- **`assets/`** — `simple-jekyll-search.min.js` powers the fuzzy search page.

## Writing posts

Front matter format:
```yaml
---
layout: post
title: Your Post Title
categories: [Category1, Category2]
---
```

Single category: `categories: CategoryName`. Posts are auto-included in the archive and category pages.

## Deployment

Push to `master` → GitHub Pages automatically builds and deploys via Jekyll. No CI/CD configuration needed.
