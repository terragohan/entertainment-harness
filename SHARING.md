# Sharing generated content

Entertainment Harness turns manga, comics, and books in your local library
into narrated recap videos. This document is the project's policy for
**sharing** that output — posting it to YouTube, TikTok, Instagram, Discord,
or anywhere else other people can see it. It applies in addition to the
software license ([GPL-3.0](LICENSE) + [Commons Clause](COMMONS-CLAUSE.md)),
which covers the code, not the videos you make with it.

## The short version

If you share a video made with Entertainment Harness, you must credit **both**:

1. **The source material** — the work's title and author (and publisher or
   official release when one exists), and the scanlation or translation group
   if the pages came from one.
2. **The tool** — "Made with Entertainment Harness (https://terragohan.com)".

Every rendered video ends with a credits card carrying exactly this
attribution. Keep it. Do not trim it, crop it out, or cover it.

## Required attribution

### In the video itself

The end card is burned into every render by default (work title, author when
known, source, chapter, and the tool). It is on by default and re-renders are
cheap, so there is no reason to strip it from anything you share. If you
disable it for private viewing (`[video] credits = false` in config.toml) or
edit it out of a cut you share, the description attribution below is
**required**, not optional.

### In the post / video description

Copy this template and fill it in:

```
<source title> by <author> — chapter(s) <range> recap.
Official release: <link to buy/read, if one exists>
[Scanlation by <group> — <link>, if applicable]
Made with Entertainment Harness — https://terragohan.com
```

Worked example:

```
Kenja no Mago by Tsuyoshi Yoshioka & Seiji Kikuchi — chapters 47–50 recap.
Official release: https://kodansha.us/series/wise-mans-grandchild/
Scanlation by <group name>.
Made with Entertainment Harness — https://terragohan.com
```

Platform notes:

- **YouTube**: the description block above; also name the work in the title
  or first pinned comment.
- **TikTok / Reels / Shorts**: put the credit in the caption; if the caption
  is too short, "Kenja no Mago ch. 47 · Made with Entertainment Harness" is
  the minimum, with the full block in a pinned comment.
- **Discord / forums / Reddit**: the description block in the post body.

## The honest fine print

**Why this is a policy and not a license clause.** Copyright in a generated
video belongs to the source work's rightsholders and (for your contributions)
to you — the tool's authors have no copyright claim over your output, so the
software license cannot legally attach conditions to it. What we *can* do is
(a) ask you to follow these terms as a condition of using the software, and
(b) make attribution the default in the output itself, which is what the
credits card does. If you choose to share, you are accountable for sharing
with attribution.

**The source material is not yours.** A recap video made from scanned pages
is a derivative work of copyrighted material. Attribution does not make
sharing legal — it is the minimum courtesy, not a defense. Uploads of
copyrighted works can draw Content ID claims, takedowns, or account strikes
regardless of narration or transformation, and fair use varies by
jurisdiction. The safest things to share are videos of material you have
rights to: your own work, public-domain or openly licensed comics and books,
or works whose rightsholders permit it.

**Respect scanlation-community norms.** Many groups forbid re-hosting and
monetization of their releases. If the group credits a page with "do not
re-upload," that applies to videos made from those pages too.

**Honor takedowns.** If an author, publisher, or scanlation group asks you
to remove a video, remove it promptly — attribution or not.

## For rightsholders

If you find content made with Entertainment Harness that infringes your
work, the fastest remedy is the platform's own takedown process. You can
also open an issue at
<https://github.com/terragohan/entertainment-harness/issues> and we will do
what we can — but note that the software is local-first: output is made on
users' own machines and is not hosted by us.
