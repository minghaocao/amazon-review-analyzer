"""Build the lightweight, unsigned Amazon Review Analyzer macOS wrapper.

The bundle contains no Python runtime, virtual environment, model, review
data, credentials, or business outputs.  It is a Finder-friendly entry point
that delegates all launch decisions to :mod:`mac_launcher`.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import stat
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path


APP_NAME = "Amazon Review Analyzer"
APP_VERSION = "1.0.1"
BUNDLE_IDENTIFIER = "com.caominghao.amazonreviewanalyzer"
BUNDLE_VERSION = "101"
APP_BUNDLE_NAME = f"{APP_NAME}.app"
EXECUTABLE_NAME = APP_NAME
ICON_NAME = "AppIcon.icns"
LOCATOR_SCHEMA = "step16.5.project-locator.v1"

ICON_SIZES: tuple[tuple[str, int], ...] = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


class AppBuildError(RuntimeError):
    """Raised when the local macOS wrapper cannot be built safely."""


def _run(command: list[str]) -> None:
    """Run one local build tool without a shell or network access."""

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AppBuildError(f"Local build tool failed: {Path(command[0]).name}") from exc
    if completed.returncode != 0:
        raise AppBuildError(f"Local build tool failed: {Path(command[0]).name}")


def _write_info_plist(destination: Path) -> None:
    payload = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": EXECUTABLE_NAME,
        "CFBundleIconFile": "AppIcon",
        "CFBundleIdentifier": BUNDLE_IDENTIFIER,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": BUNDLE_VERSION,
        "LSMinimumSystemVersion": "12.0",
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
    }
    with destination.open("wb") as handle:
        plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=True)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _inside_rounded_rectangle(
    x: int,
    y: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    radius: int,
) -> bool:
    if left + radius <= x <= right - radius or top + radius <= y <= bottom - radius:
        return left <= x <= right and top <= y <= bottom
    center_x = left + radius if x < left + radius else right - radius
    center_y = top + radius if y < top + radius else bottom - radius
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def _write_base_icon(destination: Path, size: int = 1024) -> None:
    """Draw a project-owned icon with only the Python standard library."""

    rows: list[bytes] = []
    scale = size / 1024
    for y in range(size):
        row = bytearray([0])  # PNG filter type 0.
        source_y = int(y / scale)
        for x in range(size):
            source_x = int(x / scale)
            rgba = (0, 0, 0, 0)
            if _inside_rounded_rectangle(source_x, source_y, 52, 52, 972, 972, 210):
                progress = min(1.0, max(0.0, (source_x + source_y - 100) / 1844))
                rgba = (
                    int(23 + (42 - 23) * progress),
                    int(63 + (166 - 63) * progress),
                    int(114 + (184 - 114) * progress),
                    255,
                )

            in_bubble = _inside_rounded_rectangle(
                source_x, source_y, 190, 180, 834, 690, 116
            )
            in_tail = (
                316 <= source_x <= 466
                and 630 <= source_y <= 806
                and source_y <= 806 - (source_x - 316) * 0.78
            )
            if in_bubble or in_tail:
                rgba = (247, 252, 255, 250)

            for left, top, right, bottom, color in (
                (316, 469, 386, 589, (42, 166, 184, 255)),
                (430, 389, 500, 589, (33, 103, 165, 255)),
                (544, 319, 614, 589, (23, 63, 114, 255)),
            ):
                if _inside_rounded_rectangle(
                    source_x, source_y, left, top, right, bottom, 28
                ):
                    rgba = color

            distance_squared = (source_x - 666) ** 2 + (source_y - 522) ** 2
            if 88**2 <= distance_squared <= 132**2:
                rgba = (23, 63, 114, 255)
            # Rounded diagonal magnifying-glass handle.
            handle_x = source_x - 744
            handle_y = source_y - 602
            projection = max(0.0, min(1.0, (handle_x + handle_y) / 182))
            closest_x = 91 * projection
            closest_y = 91 * projection
            if (handle_x - closest_x) ** 2 + (handle_y - closest_y) ** 2 <= 24**2:
                rgba = (23, 63, 114, 255)
            row.extend(rgba)
        rows.append(bytes(row))

    raw = b"".join(rows)
    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    destination.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _build_icon(svg_source: Path, destination: Path, work_directory: Path) -> None:
    sips = Path("/usr/bin/sips")
    iconutil = Path("/usr/bin/iconutil")
    tiffutil = Path("/usr/bin/tiffutil")
    tiff2icns = Path("/usr/bin/tiff2icns")
    if not sips.is_file() or not iconutil.is_file():
        raise AppBuildError("Required macOS icon tools are unavailable.")

    # Keep the editable SVG as the design source, while rasterization stays
    # deterministic and dependency-free for developer machines.
    if not svg_source.is_file():
        raise AppBuildError("The local icon source is unavailable.")
    base_png = work_directory / "AppIcon-1024.png"
    _write_base_icon(base_png)
    iconset = work_directory / "AppIcon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for filename, pixels in ICON_SIZES:
        _run(
            [
                str(sips),
                "-s",
                "format",
                "png",
                "-z",
                str(pixels),
                str(pixels),
                str(base_png),
                "--out",
                str(iconset / filename),
            ]
        )
    try:
        _run([str(iconutil), "-c", "icns", str(iconset), "-o", str(destination)])
    except AppBuildError:
        # Some macOS SDK combinations reject otherwise valid legacy iconsets.
        # The native TIFF converter provides a compatible, still-local path.
        if not tiffutil.is_file() or not tiff2icns.is_file():
            raise
        tiff_directory = work_directory / "AppIcon.tiffs"
        tiff_directory.mkdir(parents=True, exist_ok=True)
        tiffs: list[str] = []
        for filename, _pixels in ICON_SIZES:
            tiff = tiff_directory / f"{Path(filename).stem}.tiff"
            _run(
                [
                    str(sips),
                    "-s",
                    "format",
                    "tiff",
                    str(iconset / filename),
                    "--out",
                    str(tiff),
                ]
            )
            tiffs.append(str(tiff))
        combined = work_directory / "AppIcon.tiff"
        _run([str(tiffutil), "-cat", *tiffs, "-out", str(combined)])
        _run([str(tiff2icns), str(combined), str(destination)])
    if not destination.is_file() or destination.stat().st_size < 128:
        raise AppBuildError("The macOS application icon was not created.")


def _validate_sources(project_root: Path) -> tuple[Path, Path, Path]:
    helper = project_root / "mac_launcher.py"
    bootstrap = project_root / "launcher" / "app_bootstrap.zsh"
    icon_source = project_root / "launcher" / "assets" / "app_icon.svg"
    missing = [path.name for path in (helper, bootstrap, icon_source) if not path.is_file()]
    if missing:
        raise AppBuildError("Required launcher source is missing.")
    return helper, bootstrap, icon_source


def _copy_bundle_contents(project_root: Path, bundle: Path, scratch: Path) -> None:
    helper, bootstrap, icon_source = _validate_sources(project_root)
    contents = bundle / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    _write_info_plist(contents / "Info.plist")
    executable = macos / EXECUTABLE_NAME
    shutil.copyfile(bootstrap, executable)
    executable.chmod(
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    shutil.copyfile(helper, resources / "mac_launcher.py")
    locator = {
        "schema_version": LOCATOR_SCHEMA,
        # Resolved from the .app itself: <project>/dist/<app>.app -> <project>.
        "relative_project_path": "../..",
    }
    (resources / "project_locator.json").write_text(
        json.dumps(locator, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _build_icon(icon_source, resources / ICON_NAME, scratch)


def build_app(project_root: str | Path) -> Path:
    """Build and return ``dist/Amazon Review Analyzer.app`` atomically."""

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise AppBuildError("Project root is unavailable.")
    _validate_sources(root)

    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ara-app-build-", dir=dist) as temporary:
        scratch = Path(temporary)
        staged_bundle = scratch / APP_BUNDLE_NAME
        _copy_bundle_contents(root, staged_bundle, scratch)

        destination = dist / APP_BUNDLE_NAME
        old_bundle = scratch / "previous.app"
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise AppBuildError("Existing app destination is unsafe.")
            os.replace(destination, old_bundle)
        try:
            os.replace(staged_bundle, destination)
        except Exception:
            if old_bundle.exists() and not destination.exists():
                os.replace(old_bundle, destination)
            raise
        if old_bundle.exists():
            shutil.rmtree(old_bundle)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local macOS launcher app.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Amazon Review Analyzer project root",
    )
    return parser.parse_args()


def main() -> int:
    try:
        destination = build_app(_parse_args().project_root)
    except AppBuildError as exc:
        print(f"App build failed: {exc}")
        return 1
    print(f"Built: {destination}")
    print("Signing: unsigned development build")
    print("Notarization: not notarized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

