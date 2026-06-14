#!/usr/bin/env python3
"""Convert old element-form crs: crop tags in XMP sidecars to inline attribute form.

Usage:
    python convert_xmp_to_attr_form.py <folder> [--dry-run]

Scans <folder> (non-recursively) for *.xmp files that contain element-form crop
tags such as <crs:CropLeft>0.1</crs:CropLeft> and rewrites them as inline
attributes on rdf:Description, which is the form Lightroom expects.

Any XMP already in attribute form, or without crs: crop tags, is left unchanged.
"""

import argparse
import re
import sys
from pathlib import Path

CROP_TAGS = (
    'HasCrop', 'CropLeft', 'CropTop', 'CropRight', 'CropBottom', 'CropAngle',
    'CropConstrainToWarp', 'CropConstrainToUnitSquare',
)

_ELEMENT_RE = re.compile(
    r'\s*<crs:(' + '|'.join(CROP_TAGS) + r')>(.*?)</crs:\1>',
    re.DOTALL,
)


def _needs_conversion(content: str) -> bool:
    return bool(_ELEMENT_RE.search(content))


def _inject_attr(content: str, attrs_str: str) -> str:
    m = re.search(r'(<rdf:Description\b[^>]*)(>)', content, re.DOTALL)
    if not m:
        return content
    return content[:m.start(2)] + '\n   ' + attrs_str + content[m.start(2):]


def convert(content: str) -> str:
    # Collect element-form values in order of appearance.
    found = {}
    for m in _ELEMENT_RE.finditer(content):
        tag, value = m.group(1), m.group(2).strip()
        if tag not in found:
            found[tag] = value

    if not found:
        return content

    # Strip element-form tags.
    content = _ELEMENT_RE.sub('', content)

    # Also strip any pre-existing attribute-form crop tags to avoid duplicates.
    for tag in CROP_TAGS:
        content = re.sub(rf'\s*crs:{tag}="[^"]*"', '', content)

    # Ensure xmlns:crs is declared on rdf:Description.
    if 'xmlns:crs=' not in content:
        content = _inject_attr(
            content, 'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"')

    # Inject collected values as attributes, preserving original order.
    attrs_str = '\n   '.join(f'crs:{tag}="{val}"' for tag, val in found.items())
    content = _inject_attr(content, attrs_str)

    return content


def process_folder(folder: Path, dry_run: bool) -> None:
    xmp_files = sorted(folder.glob('*.xmp'))
    if not xmp_files:
        print(f"No .xmp files found in {folder}")
        return

    converted = 0
    skipped = 0
    for xmp in xmp_files:
        try:
            content = xmp.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            print(f"  ERROR reading {xmp.name}: {exc}", file=sys.stderr)
            continue

        if not _needs_conversion(content):
            skipped += 1
            continue

        new_content = convert(content)
        if dry_run:
            print(f"  [dry-run] would convert: {xmp.name}")
        else:
            xmp.write_text(new_content, encoding='utf-8')
            print(f"  Converted: {xmp.name}")
        converted += 1

    print(f"\n{'Would convert' if dry_run else 'Converted'} {converted} file(s), "
          f"skipped {skipped} (already correct or no crop tags).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path, help='Folder containing .xmp files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without writing files')
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory.", file=sys.stderr)
        sys.exit(1)

    process_folder(args.folder, args.dry_run)


if __name__ == '__main__':
    main()
