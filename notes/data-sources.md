# Books corpus — sources & licenses

Tier 1 — fully open, high quality, priority ingest:

| Source | License | Format | Notes |
|---|---|---|---|
| OpenStax (Calc 1/2/3, Linear Algebra, College Algebra, Statistics, Precalc) | CC-BY 4.0 | HTML + LaTeX | Clean, well-structured. Scrape from openstax.org. |
| AIM Open Textbook Initiative | various CC | PMLbook / LaTeX | ~60 vetted titles. Judson Abstract Algebra, Beezer Linear Algebra, Trench Real Analysis, Hammack Book of Proof, Keisler Elementary Calculus, Sundstrom Mathematical Reasoning. |
| Stein & Shakarchi Princeton Lectures | author-hosted PDFs | PDF | Real/Complex/Functional/Fourier. Not strictly open; flag. |
| ProofWiki full dump | CC-BY-SA 4.0 | wikitext | ~30k proofs, extremely high signal. |
| Wikipedia — Mathematics portal pages | CC-BY-SA 4.0 | wikitext | Filter by Category:Mathematics tree. |
| PlanetMath | CC-BY-SA | XML dump | Smaller but dense. |

Tier 2 — review before ingesting:

| Source | License | Notes |
|---|---|---|
| LibreTexts math | CC-BY-NC-SA | NC clause — OK for research model, flag if redistributing. |
| nLab | CC-BY-SA | Category theory, very advanced. |

Explicitly excluded from v1 (books phase):
- arXiv — papers, not books. Next phase.
- OpenWebMath — web scrape, not books. Next phase.
- LibGen / Sci-Hub — copyright.

## Quality filters
- Strip navigation, footers, "Exercise 3.14" answer-key headers.
- Preserve LaTeX `$...$` and `\begin{equation}...\end{equation}` verbatim.
- Drop pages with <200 tokens after cleanup.
- Dedup with MinHash at paragraph level.
- Language filter: English only for v1 (FastText lid).

## Expected size
Rough back-of-envelope for books-only:
- OpenStax: ~15 MB text
- AIM titles (~60): ~150 MB
- ProofWiki: ~200 MB
- Wikipedia math: ~300 MB
- PlanetMath: ~50 MB

Total ≈ 700 MB–1 GB cleaned text ≈ 200–300M tokens. Not enough to Chinchilla-train 500M from scratch (need ~10B), but enough to validate pipeline, tokenizer, and see whether text quality is the bottleneck before scaling.
