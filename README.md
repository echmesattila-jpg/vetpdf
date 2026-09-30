# VetPDF

VetPDF is a conservative veterinary/scientific PDF metadata processor. It extracts text locally, uses the OpenAI API to infer bibliographic and professional metadata from the PDF itself, and can write verified PDF/XMP metadata with ExifTool.

## Safety first

VetPDF runs in **dry-run mode by default**. It does not modify, move, rename, or overwrite PDFs unless `--write` is explicitly supplied. Low-confidence results are flagged for review and are not written automatically.

## What it does

- Reads the first pages locally with `pdftotext`.
- Falls back to vision for scanned/image-only PDFs by rendering pages locally.
- Extracts title, author surnames, publication, DOI, publication year and professional keywords.
- Uses only the supplied PDF content for AI metadata inference; no web/DOI/PubMed lookup.
- Detects possible duplicates conservatively.
- Reads PDF annotations locally.
- Optionally routes PDFs matching a configured annotation author into `annotated/`.
- Writes both classic PDF Info and XMP metadata with ExifTool.
- Verifies metadata and page count after writing.
- Creates a JSONL audit log.

## Requirements

- Python 3.10+
- OpenAI Python package
- `qpdf`
- Poppler (`pdftotext` and `pdftoppm`)
- `exiftool` (required only for `--write`)
- An OpenAI API key

### macOS with Homebrew

```bash
brew install qpdf poppler exiftool
python3 -m pip install -r requirements.txt
```

## API key

Use your own OpenAI API key. Never put it in this repository.

```bash
export OPENAI_API_KEY='your-key-here'
```

## Expected folder layout

VetPDF can be given either a collection root containing year folders or a year folder directly:

```text
MyPDFs/
├── 2019/
│   ├── 2019_Author_Title.pdf
│   └── ...
├── 2020/
└── ...
```

PDF filenames must begin with a verified four-digit publication year followed by `_`, for example:

```text
2019_Smith_Example.pdf
```

The filename year is treated as authoritative. If the visible PDF content shows a conflicting year, VetPDF flags the file for review.

## Run a dry-run

```bash
python3 vetpdf.py /path/to/MyPDFs
```

Test only the first five PDFs:

```bash
python3 vetpdf.py /path/to/MyPDFs --limit 5
```

## Write metadata

After reviewing the dry-run:

```bash
python3 vetpdf.py /path/to/MyPDFs --write
```

A completely successful full-year write renames the processed `YYYY` folder to `xYYYY` as a checkpoint. Reviews/errors prevent the year from being closed.

## Optional annotation-author routing

By default, no person's annotations receive special treatment. To route PDFs whose annotation author exactly matches a chosen name into `YEAR/annotated/`:

```bash
python3 vetpdf.py /path/to/MyPDFs --annotation-author "Example Name"
```

or:

```bash
export VETPDF_ANNOTATION_AUTHOR='Example Name'
python3 vetpdf.py /path/to/MyPDFs
```

Use `--write` only after confirming the dry-run placement decisions.

## Model

The default model is configured in `vetpdf.py`. You can override it:

```bash
python3 vetpdf.py /path/to/MyPDFs --model MODEL_NAME
```

API usage may incur charges on the user's OpenAI account.

## Privacy

Text extracted from PDFs, or rendered page images for scanned PDFs, is sent to the configured OpenAI API model for analysis. Local tools also inspect and, in write mode, modify PDF metadata. Review your organization's privacy and data-handling requirements before processing confidential documents.

## License

See `LICENSE`.
