---
layout: post
title: "Working from the terminal - Part 2"
categories: [neovim, tmux, Terminal, Gruvbox, Productivity]
---

In [Part 1]({{ site.baseurl }}{% post_url 2026-08-17-working-from-terminal-pt1 %}) I got the foundation working: tmux, Neovim, and lazygit all talking to each other with one consistent set of keybindings. This time I wanted to make the whole thing look the way I actually want to stare at for hours, a consistent Gruvbox Light theme across the entire stack.

![Terminal, tmux, and Neovim themed in Gruvbox Light](/images/terminal_blog_cover_pt2.png)

On VSCode I used Solarized Light, a warm, comfortable theme, easy to read for people who, like me, suffer astigmatism. Gruvbox Light is the closest equivalent in the Neovim world, and I'd already used it in other places, so it made sense to standardize on it everywhere.

"Everywhere" turned out to be four separate, independent layers that don't automatically sync with each other, worth calling out explicitly since it's easy to assume changing one changes them all:

1. **Neovim** itself, via the `gruvbox.nvim` colorscheme.
2. **tmux's status bar**, the bar at the bottom showing session/window names.
3. **Terminal.app**, the actual window background, text, cursor, and ANSI colors.
4. **Claude Code's own UI theme**, which turned out to be completely separate from all of the above.

## Neovim: gruvbox.nvim

Added the same way as every other plugin in this fork, via `vim.pack` (see Part 1 if `vim.pack` is new to you):

```lua
vim.pack.add { gh 'ellisonleao/gruvbox.nvim' }
```

Applied with:

```lua
vim.o.background = 'light'
vim.cmd.colorscheme 'gruvbox'
```

To preview without restarting, `:colorscheme gruvbox`. To fully reload the plugin, `:lua vim.pack.update()`.

## tmux status bar: manual styling first, then doing it properly with TPM

**First attempt**, manual: I hardcoded Gruvbox hex values directly into the `status-bar` options in `~/.tmux.conf`. It worked, but it wasn't the "proper" way, no plugin manager involved, so nothing to update, and every value hand-maintained.

So I switched to TPM (Tmux Plugin Manager) and used a real theme plugin instead.

Install TPM:

```bash
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
```

Add the plugin declarations to `~/.tmux.conf`, near the other settings:

```bash
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'egel/tmux-gruvbox'
set -g @tmux-gruvbox 'light'
```

And, as the very last lines of the file, TPM requires this:

```bash
run '~/.tmux/plugins/tpm/tpm'
```

Reload the config, then, from inside a tmux session, install the plugins with TPM's own keybind:

```
Ctrl+a then Shift+I
```

(capital `I`, TPM's install keybind.)

Once `tmux-gruvbox` was confirmed working, I deleted the earlier manual status-bar styling lines, they'd otherwise conflict with the plugin's own styling.

One scope note worth being explicit about: `tmux-gruvbox` only styles the tmux status bar. It has no effect on the terminal application's own background color, or on the shell prompt theme. Those are separate layers, next up.

## Terminal.app: no built-in profile was close enough

After tmux and Neovim were both Gruvbox Light, the actual Terminal.app window was still black, because it was running Terminal's built-in "Pro" profile, which is unrelated to both tmux and Neovim theming entirely.

I checked whether a built-in profile would get me close enough. Terminal.app ships only: Basic, Grass, Homebrew, Man Page, Novel, Ocean, Pro, Red Sands, Silver Aerogel. Novel was the closest, light, warm/cream background, but on inspection it didn't match Gruvbox's specific palette well enough.

So I built a custom profile manually:

1. Terminal → Settings → Profiles → `+` to duplicate a profile, renamed it **Gruvbox Light**.
2. Text tab, set background, foreground, cursor, and selection colors.
3. Same tab, set all 16 ANSI colors (normal + bright) to Gruvbox Light values.
4. Set the new profile as Default.

### Issue: no hex input field visible in the color picker

The macOS color picker doesn't show a hex field by default. Fix: click the second tab icon (the sliders icon) to open "Color Sliders", then make sure the dropdown is set to **RGB Sliders**, a hex field appears there directly on modern macOS. On older macOS versions without a hex field, you'll need to enter RGB (0–255) values manually instead.

### Gruvbox Light color reference (hard contrast)

| Element | Hex |
|---|---|
| Background | `#f9f5d7` |
| Foreground / text | `#3c3836` |
| Selection | `#d5c4a1` |

| ANSI | Normal | Bright |
|---|---|---|
| Black | `#fbf1c7` | `#928374` |
| Red | `#9d0006` | `#cc241d` |
| Green | `#79740e` | `#98971a` |
| Yellow | `#b57614` | `#d79921` |
| Blue | `#076678` | `#458588` |
| Magenta | `#8f3f71` | `#b16286` |
| Cyan | `#427b58` | `#689d6a` |
| White | `#3c3836` | `#7c6f64` |

## Claude Code: still showing a black background on message submit

Symptom: with Terminal, tmux, and Neovim all matching Gruvbox Light, Claude Code's own interface, running inside that same terminal, still flashed a black background every time I sent a message.

Cause: Claude Code has its own internal `/theme` setting, completely separate from the terminal's color profile, tmux, or Neovim. It defaults to dark and doesn't automatically detect or follow the terminal's background color unless you tell it to.

Fix, from inside a Claude Code session:

```
/theme
```

This opens an interactive picker, select `light`, or `auto` to have it detect the terminal's light/dark background and follow OS-level appearance changes automatically going forward.

One known limitation, not fixable from terminal config: per Claude Code's own documentation, some UI elements, code block backgrounds, diff highlighting, and yes/no prompt selection highlights, currently use hardcoded colors that don't fully respect a custom terminal palette. So a few accents may still look slightly off from true Gruvbox even with `/theme light` set. That's a known constraint on Anthropic's side, not a local configuration issue.

## Wrapping up

The main takeaway: "theming my terminal" isn't one setting, it's four independent ones (Neovim, tmux, the terminal emulator, and any TUI apps running inside it, like Claude Code), and each needs to be told about the theme separately. Worth checking all of them individually rather than assuming one change propagates everywhere.

With this done, Terminal, tmux, Neovim, and Claude Code all now share the same Gruvbox Light look, consistent, warm, and comfortable to stare at for long stretches.

## Versions

- **macOS**: 26.5
- **tmux**: 3.7b
- **Neovim**: v0.13.0-dev
- **gruvbox.nvim**: latest via `vim.pack`
- **tmux-gruvbox**: latest via TPM
- **Claude Code**: latest, with `/theme` support
