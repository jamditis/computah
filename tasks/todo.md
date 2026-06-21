# todo

## plan

- [x] Inspect repository instructions and current docs.
- [x] Expand README with clearer pitch, architecture, setup, configuration, testing, and contribution notes.
- [x] Add a designed GitHub Pages site with matching favicon and social metadata.
- [x] Run format/static checks and available tests.
- [x] Review the full diff and record results.

## review

- README was expanded with setup, configuration, testing, roadmap, and limitation notes.
- GitHub Pages was added under `docs/` with a matching SVG favicon and social preview asset.
- Full pipeline tests are blocked in this checkout until the Piper voice model is downloaded.

## review follow-ups (PR #1)

- Normalized README headings, table headers, and list items to sentence case; kept product names (GitHub, GitHub Pages) in official casing.
- Closed an unclosed `.links` div in `docs/index.html` that left the nav markup malformed.
- Replaced the SVG `og:image`/`twitter:image` references with a rasterized 1200x630 `social-card.png`, since Twitter and most Open Graph consumers do not render SVG previews; added `og:image:width`, `og:image:height`, and `og:image:type`. The SVG is kept as the source for the PNG.
