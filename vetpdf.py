#!/usr/bin/env python3
"""
vetpdf.py — Veterinary PDF metadata processor
Version 1.0.0 — PUBLIC EDITION — SAFE DRY-RUN DEFAULT

Default behavior:
- Accepts a PDF collection root or a YYYY folder; if given a root, finds the earliest YYYY folder
- Reads PDF text locally with pdftotext; image-only PDFs fall back to OpenAI vision from locally rendered pages
- Asks OpenAI to infer bibliographic/professional metadata ONLY from extracted PDF text
- Checks PDF annotations locally with qpdf and can optionally match a configured annotation author
- Detects likely duplicate articles within the selected year
- Writes a JSONL audit log
- DOES NOT modify, move, rename, or overwrite PDFs unless --write is explicitly used

Write mode (later, after dry-run validation):
- Writes Title/Author/Subject/Keywords with ExifTool
- Verifies PDF/page count/metadata
- Optionally moves PDFs annotated by a configured author into YEAR/annotated/
- Renames YEAR -> xYEAR only if the entire year succeeds
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: Python package 'openai' is missing.")
    print("Install with: python3 -m pip install --user openai")
    sys.exit(2)

MODEL_DEFAULT = "gpt-5.6-sol"

SYSTEM_PROMPT = r"""
You analyze veterinary/scientific PDF content for bibliographic metadata.

STRICT SOURCE RULE:
Use ONLY the PDF text supplied in this request. Do not use web search, outside
knowledge, DOI lookup, Crossref, PubMed, Google Scholar, or assumptions based
on the filename.

Determine:
- title: full true publication title
- authors: citation-style FAMILY NAMES / SURNAMES ONLY, for ALL authors in publication order. Do not include given names, initials, degrees, credentials, or titles. Preserve compound family names when supported by the PDF (example: "Luis Fuentes" stays "Luis Fuentes").
- subject: full journal/publication name; DO NOT append the year
- publication_year_in_pdf: publication year only if explicitly visible in the supplied PDF text; otherwise empty string. The authoritative publication year is supplied separately from the verified filename.
- doi: DOI only if explicitly present in the supplied PDF text and clearly
  belongs to this publication; otherwise empty string
- keywords: semicolon-separated professional topics based on actual article
  content. Use the publication's own Keywords/Key words, abstract/summary, and
  the supplied relevant article text. Include all genuinely relevant veterinary
  specialties, diseases, treatments, procedures, technologies, species, drugs,
  toxins, diagnostics, clinical states, etc. Do not add a field merely because
  one parameter occurs in a table/scoring system.
- confidence: "high", "medium", or "low"
- review_reason: empty if reliable; otherwise a concise explanation

For authors, return only the family name used for citation. Never expand names from outside knowledge and never use "et al.". If the family name cannot be determined reliably from the supplied PDF, keep the supported author form and set a review_reason rather than guessing.
Do not invent missing data.

Return ONLY valid JSON with exactly these keys:
{
  "title": "...",
  "authors": ["...", "..."],
  "subject": "...",
  "publication_year_in_pdf": "YYYY",
  "doi": "...",
  "keywords": "item; item; item",
  "confidence": "high|medium|low",
  "review_reason": "..."
}
"""

def run(cmd, check=True, text=True):
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=text, check=check
    )

def require_tool(name: str):
    if shutil.which(name) is None:
        print(f"ERROR: required command not found: {name}")
        sys.exit(2)

def find_year(root: Path) -> Optional[Path]:
    years = []
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"\d{4}", p.name):
            years.append(p)
    return min(years, key=lambda p: int(p.name)) if years else None

def pdfs_in_year(year_dir: Path):
    # Include existing annotated/ PDFs too, but never duplicate.
    return sorted(
        [p for p in year_dir.rglob("*.pdf") if p.is_file()],
        key=lambda p: str(p).lower()
    )

def page_count(pdf: Path) -> int:
    # qpdf --show-npages is fast and doesn't rewrite the file.
    r = run(["qpdf", "--warning-exit-0", "--show-npages", str(pdf)])
    return int(r.stdout.strip())

def extract_text(pdf: Path, max_pages=8, max_chars=45000) -> str:
    """
    Extract first pages locally. 8 pages is intentionally conservative:
    usually enough for title/authors/abstract/keywords while avoiding full-paper
    processing. If metadata is uncertain, the file is flagged for review rather
    than silently reading/guessing everything.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "text.txt"
        cmd = [
            "pdftotext", "-f", "1", "-l", str(max_pages),
            "-layout", "-enc", "UTF-8", str(pdf), str(out)
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"pdftotext failed: {r.stderr.strip()}")
        text = out.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]

def render_pdf_pages_for_vision(pdf: Path, max_pages=8, dpi=150):
    """
    Render PDF pages locally to JPEG for vision fallback.
    Nothing is written beside the original PDF.
    Returns (TemporaryDirectory, [Path, ...]); caller must cleanup td.
    """
    td = tempfile.TemporaryDirectory()
    prefix = Path(td.name) / "page"
    cmd = [
        "pdftoppm", "-f", "1", "-l", str(max_pages),
        "-jpeg", "-r", str(dpi), "-jpegopt", "quality=82",
        str(pdf), str(prefix)
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        td.cleanup()
        raise RuntimeError(f"PDF page rendering failed: {r.stderr.strip()}")
    pages = sorted(Path(td.name).glob("page-*.jpg"))
    if not pages:
        td.cleanup()
        raise RuntimeError("PDF page rendering produced no images.")
    return td, pages

def analyze_with_vision(client: OpenAI, filename: str, pages, model: str):
    """
    Image-only/scanned PDF fallback. Uses rendered page images, not OCR.
    Same strict source rule: only the supplied PDF page images.
    """
    import base64

    verified_year = verified_year_from_filename(filename)
    prompt = (
        f"FILENAME: {filename}\n"
        f"AUTHORITATIVE VERIFIED PUBLICATION YEAR FROM FILENAME: {verified_year}\n"
        "The year above is trusted. Analyze ONLY the attached page images from this PDF. "
        "Do not use web search, DOI lookup, outside knowledge, or assumptions from the filename. "
        "Do not lower confidence merely because the publication year is absent from the visible pages. "
        "If the PDF visibly shows a different publication year, put it in publication_year_in_pdf "
        "and explain the conflict in review_reason.\n\n"
        "Return ONLY the same JSON object required by the system instructions."
    )
    content = [{"type": "input_text", "text": prompt}]
    for page in pages:
        b64 = base64.b64encode(page.read_bytes()).decode("ascii")
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
            "detail": "high",
        })

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": content}],
    )
    raw = response.output_text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return normalize_ai_metadata(data, filename)

# Deterministic, locally maintained publication abbreviation dictionary.
# Only unambiguous aliases belong here. Unknown abbreviations are NEVER guessed.
PUBLICATION_ALIASES = {
    "javma": "Journal of the American Veterinary Medical Association",
    "j am vet med assoc": "Journal of the American Veterinary Medical Association",
    "jvecc": "Journal of Veterinary Emergency and Critical Care",
    "j vet emerg crit care": "Journal of Veterinary Emergency and Critical Care",
    "jvim": "Journal of Veterinary Internal Medicine",
    "j vet intern med": "Journal of Veterinary Internal Medicine",
    "jsap": "Journal of Small Animal Practice",
    "j small anim pract": "Journal of Small Animal Practice",
    "jfms": "Journal of Feline Medicine and Surgery",
    "j feline med surg": "Journal of Feline Medicine and Surgery",
    "vet clin north am small anim pract": "Veterinary Clinics of North America: Small Animal Practice",
    "nephrol dial transplant": "Nephrology Dialysis Transplantation",
}

def expand_publication_name(data):
    """Expand only known, unambiguous journal/publication abbreviations."""
    subject = str(data.get("subject", "") or "").strip()
    key = re.sub(r"[.]+", "", subject).casefold().strip()
    full = PUBLICATION_ALIASES.get(key)
    if not full:
        return data

    data["subject"] = full

    # If the ONLY reason for REVIEW was that this known publication was shown
    # as an abbreviation/acronym, the deterministic dictionary resolves it.
    # Never clear a review that also concerns year/authors/title/etc.
    reason = str(data.get("review_reason", "") or "").strip()
    if reason:
        low = reason.casefold()
        abbrev_issue = (
            ("abbrevi" in low or "acronym" in low)
            and ("journal" in low or "publication" in low)
            and ("full name" in low or "full journal" in low or "full publication" in low)
        )
        other_issue_markers = (
            "year conflict", "different publication year", "author", "title", "doi",
            "cannot", "uncertain", "ambiguous", "not visible", "missing"
        )
        # "not visible/present" commonly describes the full journal name itself;
        # allow that wording when no substantive metadata issue is mentioned.
        substantive = any(x in low for x in other_issue_markers[:8])
        if abbrev_issue and not substantive:
            data["review_reason"] = ""
            if data.get("confidence") in {"low", "medium"}:
                data["confidence"] = "high"
    return data

def normalize_ai_metadata(data, filename: str):
    required = {
        "title", "authors", "subject", "publication_year_in_pdf",
        "doi", "keywords", "confidence", "review_reason"
    }
    if set(data.keys()) != required:
        raise ValueError(f"Unexpected JSON keys: {sorted(data.keys())}")
    if not isinstance(data["authors"], list):
        raise ValueError("authors must be a list")

    data["authors"] = strip_author_degrees(data["authors"])
    data = expand_publication_name(data)
    verified_year = verified_year_from_filename(filename)
    pdf_year = str(data.get("publication_year_in_pdf", "") or "").strip()
    data["publication_year"] = verified_year
    if pdf_year and pdf_year != verified_year:
        data["confidence"] = "low"
        conflict = f"YEAR CONFLICT: verified filename year={verified_year}, PDF text year={pdf_year}"
        data["review_reason"] = conflict + (("; " + data["review_reason"]) if data["review_reason"] else "")
    elif data.get("review_reason") and "year" in data["review_reason"].lower() and not pdf_year:
        data["review_reason"] = ""
        if data["confidence"] != "high":
            data["confidence"] = "high"
    return data

def duplicate_key(meta):
    """
    Conservative duplicate key: normalized title + verified year.
    Only flags; never deletes/moves based on this.
    """
    title = str(meta.get("title", "")).casefold()
    title = re.sub(r"[^a-z0-9]+", "", title)
    year = str(meta.get("publication_year", ""))
    return (year, title) if title else None

def qdf_annotation_text(pdf: Path) -> str:
    # qpdf QDF to stdout; binary-safe decode so odd PDF encodings don't crash.
    r = subprocess.run(
        ["qpdf", "--warning-exit-0", "--qdf", "--object-streams=disable", str(pdf), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if r.returncode not in (0, 2, 3):  # qpdf may use warning statuses depending on build
        raise RuntimeError(r.stderr.decode("utf-8", errors="replace"))
    return r.stdout.decode("latin-1", errors="replace")

def annotation_info(pdf: Path, annotation_author: str = ""):
    s = qdf_annotation_text(pdf)
    has_annots = "/Annots" in s

    # Extract simple literal-string /T values. This intentionally does not guess
    # identities from document metadata.
    raw_authors = re.findall(r"/T\s*\((.*?)\)", s, flags=re.S)
    authors = []
    for a in raw_authors:
        a = a.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
        a = re.sub(r"\s+", " ", a).strip()
        if a and a not in authors:
            authors.append(a)

    # Optional exact annotation-author match. Disabled unless explicitly configured.
    def is_target_author(a: str) -> bool:
        if not annotation_author:
            return False
        def norm(v: str) -> str:
            return re.sub(r"\s+", " ", v.casefold()).strip()
        return norm(a) == norm(annotation_author)


    return {
        "annotations_present": has_annots,
        "annotation_authors": authors,
        "target_author": any(is_target_author(a) for a in authors),
    }

def verified_year_from_filename(filename: str) -> str:
    m = re.match(r"^(\d{4})_", filename)
    if not m:
        raise ValueError("Filename does not begin with verified YYYY_: " + filename)
    return m.group(1)

def normalize_author_case(name: str) -> str:
    """
    Convert an author name to normal capitalization ONLY when its alphabetic
    characters are effectively all uppercase. Mixed-case names are untouched.
    Keeps initials such as J. S. and handles hyphen/apostrophe-separated parts.
    """
    name = name.strip()
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return name

    # If the whole name is uppercase, normalize all tokens.  Otherwise only
    # normalize suspicious surname/name tokens that are themselves ALL CAPS
    # (eg. "George BAYLISS" -> "George Bayliss").
    whole_upper = all((not c.islower()) for c in letters)

    def fix_token(token: str) -> str:
        # Initials/acronym-like short tokens remain uppercase: A., J.S., etc.
        alpha = "".join(c for c in token if c.isalpha())
        if len(alpha) <= 1:
            return token.upper()

        # Preserve punctuation while title-casing alphabetic runs.
        parts = re.split(r"([-'’])", token)
        out = []
        for part in parts:
            if part in {"-", "'", "’"}:
                out.append(part)
            elif part:
                # Common surname particles are lower-case unless first token;
                # conservative title-case is preferable to guessing ethnicity.
                out.append(part[:1].upper() + part[1:].lower())
        return "".join(out)

    out = []
    for tok in name.split():
        alpha = "".join(c for c in tok if c.isalpha())
        token_upper = len(alpha) > 1 and alpha.upper() == alpha
        out.append(fix_token(tok) if (whole_upper or token_upper) else tok)
    return " ".join(out)

def strip_author_degrees(authors):
    # Remove common academic/professional suffixes while preserving author names.
    degree_re = re.compile(
        r",?\s+(?:DVM|VMD|PhD|Ph\.D\.|MD|M\.D\.|MSc|MS|MPH|DACVECC|DACVIM|DACVS|"
        r"DACVAA|DECVIM-CA|DECVECC|FRCVS|BVSc|BVM&S|DrMedVet|Dr\.?\s*med\.?\s*vet\.?)"
        r"(?:\s*,\s*|\s*$)",
        re.I
    )
    cleaned = []
    for author in authors:
        a = author.strip()
        previous = None
        while previous != a:
            previous = a
            a = degree_re.sub(" ", a).strip(" ,")
        a = normalize_author_case(a)
        cleaned.append(a)
    return cleaned

def analyze_with_ai(client: OpenAI, filename: str, pdf_text: str, model: str):
    verified_year = verified_year_from_filename(filename)
    prompt = (
        f"FILENAME: {filename}\n"
        f"AUTHORITATIVE VERIFIED PUBLICATION YEAR FROM FILENAME: {verified_year}\n"
        "The year above is trusted. Do not lower confidence merely because the year is absent from PDF text. "
        "If the PDF explicitly shows a different publication year, report that in publication_year_in_pdf and review_reason.\n\n"
        "PDF TEXT START\n"
        f"{pdf_text}\n"
        "PDF TEXT END"
    )
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )
    raw = response.output_text.strip()
    # tolerate accidental fenced JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)

    return normalize_ai_metadata(data, filename)

def exiftool_read(pdf: Path):
    # Read both classic PDF Info and XMP metadata. XMP dc:Subject is the
    # standards-based keyword list used by metadata consumers such as Spotlight.
    r = run([
        "exiftool", "-j", "-G1",
        "-PDF:Title", "-PDF:Author", "-PDF:Creator", "-PDF:Subject", "-PDF:Keywords",
        "-XMP-dc:Title", "-XMP-dc:Creator", "-XMP-dc:Description", "-XMP-dc:Subject",
        "-XMP-pdf:Keywords", str(pdf)
    ])
    arr = json.loads(r.stdout)
    return arr[0] if arr else {}

def _keyword_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"\s*;\s*", str(value))
    return [str(x).strip() for x in items if str(x).strip()]

def _same_text(got, expected):
    return str(got or "").strip() == str(expected or "").strip()

def exiftool_write(pdf: Path, meta: dict):
    # Rebuild XMP from a clean slate. Some PDFs contain multiple/legacy XMP
    # packets; clearing individual dc fields can leave older arrays behind.
    # First remove XMP metadata only (PDF Info and visible PDF content remain),
    # then write one fresh XMP set plus the classic PDF Info fields.
    author = ", ".join(meta["authors"])
    keywords = _keyword_list(meta["keywords"])

    clear = run([
        "exiftool", "-m", "-overwrite_original", "-XMP:all=", str(pdf)
    ], check=False)
    if clear.returncode != 0:
        detail = (clear.stderr or clear.stdout or "ExifTool XMP clear failed").strip()
        raise RuntimeError(f"ExifTool XMP clear failed (exit {clear.returncode}): {detail}")

    cmd = [
        "exiftool", "-m", "-overwrite_original",
        f"-PDF:Title={meta['title']}",
        f"-PDF:Author={author}",
        # macOS Spotlight maps PDF Creator to kMDItemAuthors for many PDFs.
        # Keep it identical to Author so Finder shows the citation surnames.
        f"-PDF:Creator={author}",
        f"-PDF:Subject={meta['subject']}",
        f"-PDF:Keywords={meta['keywords']}",
        f"-XMP-dc:Title={meta['title']}",
        f"-XMP-dc:Description={meta['subject']}",
        f"-XMP-pdf:Keywords={meta['keywords']}",
    ]
    for a in meta["authors"]:
        cmd.append(f"-XMP-dc:Creator+={a}")
    for kw in keywords:
        cmd.append(f"-XMP-dc:Subject+={kw}")
    cmd.append(str(pdf))

    r = run(cmd, check=False)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "ExifTool write failed").strip()
        raise RuntimeError(f"ExifTool write failed (exit {r.returncode}): {detail}")
    return r.stdout.strip()

def verify_written(pdf: Path, before_pages: int, meta: dict):
    after_pages = page_count(pdf)
    if before_pages != after_pages:
        raise RuntimeError(f"Page count changed: {before_pages} -> {after_pages}")
    md = exiftool_read(pdf)

    expected_author = ", ".join(meta["authors"])
    failures = []

    # Classic PDF Info checks. Empty expected values are valid and must not fail.
    for key, expected in {
        "PDF:Title": meta["title"],
        "PDF:Author": expected_author,
        "PDF:Creator": expected_author,
        "PDF:Subject": meta["subject"],
    }.items():
        if not _same_text(md.get(key), expected):
            failures.append(f"{key}: expected={expected!r}, got={md.get(key, '')!r}")

    # ExifTool may return PDF:Keywords either as one semicolon-separated string
    # or as a JSON list. Treat both representations as equivalent.
    got_pdf_keywords = _keyword_list(md.get("PDF:Keywords"))
    exp_pdf_keywords = _keyword_list(meta["keywords"])
    if got_pdf_keywords != exp_pdf_keywords:
        failures.append(
            f"PDF:Keywords: expected={exp_pdf_keywords!r}, got={got_pdf_keywords!r}"
        )

    # XMP checks. Creator and Subject may be returned by ExifTool as lists.
    if not _same_text(md.get("XMP-dc:Title"), meta["title"]):
        failures.append(f"XMP-dc:Title mismatch: {md.get('XMP-dc:Title', '')!r}")
    if not _same_text(md.get("XMP-dc:Description"), meta["subject"]):
        failures.append(f"XMP-dc:Description mismatch: {md.get('XMP-dc:Description', '')!r}")

    got_creators = md.get("XMP-dc:Creator", [])
    if not isinstance(got_creators, list):
        got_creators = [got_creators] if got_creators else []
    got_creators = [str(x).strip() for x in got_creators if str(x).strip()]
    exp_creators = [str(x).strip() for x in meta["authors"] if str(x).strip()]
    if got_creators != exp_creators:
        failures.append(f"XMP-dc:Creator: expected={exp_creators!r}, got={got_creators!r}")

    got_keywords = _keyword_list(md.get("XMP-dc:Subject"))
    exp_keywords = _keyword_list(meta["keywords"])
    if got_keywords != exp_keywords:
        failures.append(f"XMP-dc:Subject: expected={exp_keywords!r}, got={got_keywords!r}")

    if failures:
        raise RuntimeError("; ".join(failures))
    return True

def safe_move_to_annotated(pdf: Path, year_dir: Path) -> Path:
    annotated = year_dir / "annotated"
    annotated.mkdir(exist_ok=True)
    dest = annotated / pdf.name
    if pdf.resolve() == dest.resolve():
        return pdf
    if dest.exists():
        raise RuntimeError(f"Destination already exists: {dest}")
    pdf.rename(dest)
    return dest

def safe_return_from_annotated(pdf: Path, year_dir: Path) -> Path:
    """
    If a PDF is directly inside YEAR/annotated but does not match the configured
    annotation author, return it to YEAR/. Never overwrite an existing file.
    """
    annotated = year_dir / "annotated"
    if pdf.parent.resolve() != annotated.resolve():
        return pdf
    dest = year_dir / pdf.name
    if dest.exists():
        raise RuntimeError(f"Cannot return from annotated; destination already exists: {dest}")
    pdf.rename(dest)
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", help="ZSPDF root folder")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--write", action="store_true",
                    help="Actually write metadata, optionally route matching annotated PDFs, and rename completed year")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N PDFs (ideal for testing)")
    ap.add_argument("--annotation-author", default=os.getenv("VETPDF_ANNOTATION_AUTHOR", ""),
                    help="Optional exact annotation author to route into annotated/ (or set VETPDF_ANNOTATION_AUTHOR)")
    args = ap.parse_args()

    if args.root is None:
        print("Húzd ide a feldolgozandó mappát,")
        raw_root = input("majd nyomj ENTER-t:\n\n").strip()
        # Terminal drag-and-drop may escape spaces and/or wrap the path in quotes.
        if (raw_root.startswith("'") and raw_root.endswith("'")) or (raw_root.startswith('"') and raw_root.endswith('"')):
            raw_root = raw_root[1:-1]
        raw_root = raw_root.replace("\\ ", " ")
        args.root = Path(raw_root)

    root = args.root.expanduser().resolve()
    print(f"\nMappa:\n{root}\n")
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        sys.exit(2)

    require_tool("qpdf")
    require_tool("pdftotext")
    require_tool("pdftoppm")
    if args.write:
        require_tool("exiftool")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Example for this Terminal session:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(2)

    client = OpenAI()

    # Accept either the ZSPDF/root folder OR a YEAR folder dragged directly.
    # This is especially useful for production runs one year at a time.
    if re.fullmatch(r"\d{4}", root.name):
        year_dir = root
    else:
        year_dir = find_year(root)

    if not year_dir:
        print("No non-x YEAR folder found.")
        return

    pdfs = pdfs_in_year(year_dir)
    if args.limit:
        pdfs = pdfs[:args.limit]

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"\nYEAR: {year_dir.name}")
    print(f"MODE: {mode}")
    print(f"PDFs selected: {len(pdfs)}\n")

    log_path = root / f"vetpdf_{year_dir.name}.jsonl"
    success = 0
    review = 0
    errors = 0
    target_annotation_count = 0
    duplicate_count = 0
    seen_articles = {}

    with log_path.open("a", encoding="utf-8") as log:
        for i, pdf in enumerate(pdfs, 1):
            rel = pdf.relative_to(root)
            print(f"[{i}/{len(pdfs)}] {rel}")
            record = {"file": str(rel), "mode": mode}

            try:
                before_pages = page_count(pdf)
                text = extract_text(pdf)
                extraction_mode = "text"
                if len(text.strip()) < 200:
                    print("  Text layer: insufficient → vision fallback")
                    td = None
                    try:
                        td, vision_pages = render_pdf_pages_for_vision(pdf)
                        meta = analyze_with_vision(client, pdf.name, vision_pages, args.model)
                        extraction_mode = "vision"
                    finally:
                        if td is not None:
                            td.cleanup()
                else:
                    meta = analyze_with_ai(client, pdf.name, text, args.model)

                ann = annotation_info(pdf, args.annotation_author)
                record.update({
                    "metadata": meta,
                    "annotations": ann,
                    "pages": before_pages,
                    "extraction_mode": extraction_mode
                })

                dkey = duplicate_key(meta)
                duplicate_of = seen_articles.get(dkey) if dkey else None
                if duplicate_of:
                    duplicate_count += 1
                    record["possible_duplicate_of"] = duplicate_of
                elif dkey:
                    seen_articles[dkey] = str(rel)

                if ann["target_author"]:
                    target_annotation_count += 1

                print(f"  Title: {meta['title']}")
                print(f"  Authors: {', '.join(meta['authors'])}")
                print(f"  Subject: {meta['subject']}")
                print(f"  Year: {meta['publication_year']}  DOI: {meta['doi'] or '—'}")
                print(f"  Keywords: {meta['keywords']}")
                print(f"  Confidence: {meta['confidence']}")
                print(f"  Source mode: {'VISION' if extraction_mode == 'vision' else 'TEXT'}")
                print(f"  Target annotation: {'YES' if ann['target_author'] else 'NO'}")
                annotated_dir = year_dir / "annotated"
                if not args.write:
                    if ann["target_author"] and pdf.parent.resolve() != annotated_dir.resolve():
                        print("  Placement: would move → annotated/")
                    elif (not ann["target_author"]) and pdf.parent.resolve() == annotated_dir.resolve():
                        print("  Placement: would return → YEAR/")
                    elif ann["target_author"]:
                        print("  Placement: annotated/ OK")
                    else:
                        print("  Placement: YEAR/ OK")
                if duplicate_of:
                    print(f"  ⚠ POSSIBLE DUPLICATE OF: {duplicate_of}")

                if meta["confidence"] != "high" or meta["review_reason"]:
                    review += 1
                    record["status"] = "REVIEW"
                    print(f"  ⚠ REVIEW: {meta['review_reason'] or 'AI confidence not high'}")
                    # Never write uncertain metadata automatically.
                elif args.write:
                    exiftool_write(pdf, meta)
                    verify_written(pdf, before_pages, meta)
                    final_pdf = pdf
                    if ann["target_author"]:
                        final_pdf = safe_move_to_annotated(pdf, year_dir)
                    else:
                        final_pdf = safe_return_from_annotated(pdf, year_dir)

                    # Verify any moved/returned PDF still opens and page count is unchanged.
                    if page_count(final_pdf) != before_pages:
                        raise RuntimeError("Post-move PDF page count verification failed.")
                    success += 1
                    record["status"] = "OK"
                    record["final_file"] = str(final_pdf.relative_to(root))
                    print("  Metadata verification: OK")
                    print("  File integrity: OK")
                else:
                    success += 1
                    record["status"] = "DRY-RUN-OK"
                    print("  DRY-RUN: no file changes")

            except Exception as e:
                errors += 1
                record["status"] = "ERROR"
                record["error"] = str(e)
                print(f"  ✗ ERROR: {e}")

            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print()

    print("=" * 60)
    print(f"YEAR: {year_dir.name}")
    print(f"Selected PDFs: {len(pdfs)}")
    print(f"OK / dry-run OK: {success}")
    print(f"Review required: {review}")
    print(f"Errors: {errors}")
    print(f"Target-author annotated PDFs: {target_annotation_count}")
    print(f"Possible duplicates: {duplicate_count}")
    print(f"Audit log: {log_path}")

    # Only close/rename a full year, never a limited test.
    if args.write and not args.limit and errors == 0 and review == 0 and success == len(pdfs):
        new_dir = year_dir.with_name("x" + year_dir.name)
        if new_dir.exists():
            print(f"ERROR: checkpoint destination already exists: {new_dir}")
            sys.exit(1)
        year_dir.rename(new_dir)
        if year_dir.exists() or not new_dir.exists():
            print("ERROR: folder rename verification failed.")
            sys.exit(1)
        print(f"FOLDER RENAMED: {year_dir.name} -> {new_dir.name}")
        print("FOLDER RENAME VERIFICATION: OK")
        print("STATUS: COMPLETE")
    elif args.write:
        print("YEAR NOT CLOSED: errors/reviews/limited test present.")
    else:
        print("DRY-RUN COMPLETE: originals were not changed.")

if __name__ == "__main__":
    main()
