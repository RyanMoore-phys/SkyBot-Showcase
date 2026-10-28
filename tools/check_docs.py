#!/usr/bin/env python3
"""Check controlled documentation names, registration, and local Markdown links."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CATEGORIES = {
    "governance": "gov",
    "architecture": "arc",
    "operations": "ops",
    "assurance": "ass",
    "research": "res",
    "foundation": "fnd",
    "decisions": "adr",
    "security": "sec",
}
NAME = re.compile(r"^(?P<prefix>[a-z]{3})-(?P<number>\d{3})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
LINK = re.compile(r"!??\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")
CONTROL = re.compile(
    r"^> \*\*Document control:\*\* "
    r"(?:\[(?P<link_id>[A-Z]{3}-\d{3})\]\([^)]*\)|(?P<plain_id>[A-Z]{3}-\d{3}))",
    re.MULTILINE,
)


def main() -> int:
    errors: list[str] = []
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    seen_ids: dict[str, Path] = {}
    markdown_files = [ROOT / "README.md", *DOCS.rglob("*.md")]
    canonical = [path for path in DOCS.rglob("*.md") if path != DOCS / "README.md"]

    for path in canonical:
        relative = path.relative_to(DOCS)
        # docs/legacy/<category>/<same name> holds superseded or retired records: same naming, same IDs,
        # still registered (an ID is permanent). 2026-10-22.
        parts = relative.parts[1:] if relative.parts[0] == "legacy" else relative.parts
        expected_prefix = CATEGORIES.get(parts[0]) if len(parts) > 1 else None
        match = NAME.fullmatch(path.name)
        if not match or not expected_prefix or match.group("prefix") != expected_prefix:
            errors.append(f"{path.relative_to(ROOT)}: filename does not follow the category naming protocol")
        controls = list(CONTROL.finditer(path.read_text(encoding="utf-8")))
        if len(controls) != 1:
            errors.append(f"{path.relative_to(ROOT)}: expected exactly one document control line")
            continue
        doc_id = controls[0].group("link_id") or controls[0].group("plain_id")
        if expected_prefix and doc_id != f"{expected_prefix.upper()}-{match.group('number') if match else ''}":
            errors.append(f"{path.relative_to(ROOT)}: document ID does not match filename")
        if doc_id in seen_ids:
            errors.append(f"{path.relative_to(ROOT)}: duplicate document ID {doc_id} (also {seen_ids[doc_id].relative_to(ROOT)})")
        seen_ids[doc_id] = path
        if doc_id not in index:
            errors.append(f"{path.relative_to(ROOT)}: {doc_id} is not registered in docs/README.md")

    for source in markdown_files:
        content = source.read_text(encoding="utf-8")
        for link_match in LINK.finditer(content):
            raw = link_match.group(1) or link_match.group(2) or ""
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (source.parent / unquote(parsed.path)).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                errors.append(f"{source.relative_to(ROOT)}: local link escapes repository: {raw}")
                continue
            if not target.exists():
                errors.append(f"{source.relative_to(ROOT)}: broken local link: {raw}")

    # Reciprocal references: if A links B, B links A (generated block); no document is an island.
    sys.path.insert(0, str(ROOT / "tools"))
    import doc_backlinks
    new_text, islands = doc_backlinks.compute()
    for doc_path, text in new_text.items():
        if doc_path.read_text(encoding="utf-8") != text:
            errors.append(f"{doc_path.relative_to(ROOT)}: generated 'Referenced by' block is stale "
                          "(run python3 tools/doc_backlinks.py)")
    for island in islands:
        errors.append(f"{island}: links to no other document and no document links it")

    if errors:
        print("Documentation integrity check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Documentation integrity check passed: {len(canonical)} controlled documents, {len(markdown_files)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
