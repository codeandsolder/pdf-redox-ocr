# /// script
# requires-python = ">=3.12"
# dependencies = ["pikepdf>=10.10","pillow>=11","numpy>=2","opencv-python-headless>=4.11","rapidocr>=3,<4","onnxruntime>=1.21"]
# ///
"""Source-pixel mosaic honors transformed m/l rectangle and negative re clips."""

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import math
import random
import statistics
import zlib
from unittest.mock import patch

import numpy as np
from PIL import Image
import pikepdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_image_placements import enumerate_image_paints, rectangle_path_quad, image_quad
from reconstruct_raster_mosaic import (
    composite,
    cluster_native_raster_paints,
    decode_mosaic_image,
)


def make_fixture(path):
    with pikepdf.Pdf.new() as doc:
        raw = Image.new("RGB", (120, 60), (0, 125, 215))
        img = pikepdf.Stream(doc, raw.tobytes())
        img["/Type"] = pikepdf.Name("/XObject")
        img["/Subtype"] = pikepdf.Name("/Image")
        img["/Width"] = 120
        img["/Height"] = 60
        img["/BitsPerComponent"] = 8
        img["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
        img = doc.make_indirect(img)
        resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=img))
        lines = [
            b"q 80 20 m 320 20 l 320 140 l 80 140 l h W n 300 0 0 120 40 20 cm /Im0 Do Q",
            b"q 320 140 -240 -120 re W n 300 0 0 120 40 20 cm /Im0 Do Q",
        ]
        for content in lines:
            page = doc.add_blank_page(page_size=(400, 200))
            page.obj["/Resources"] = resources
            page.Contents = pikepdf.Stream(doc, content)
        doc.save(path)


def test_clips_and_mosaic():
    assert rectangle_path_quad([[0, 0], [10, 0], [10, 6], [0, 6]]) is not None
    assert rectangle_path_quad([[0, 0], [10, 0], [9, 6], [0, 6]]) is None
    with TemporaryDirectory(prefix="pdf-redox-native-mosaic-") as td:
        path = Path(td) / "fixture.pdf"
        make_fixture(path)
        with pikepdf.Pdf.open(path) as doc:
            layouts = []
            for n in (1, 2):
                placements, images, _ = enumerate_image_paints(doc, [n])
                assert len(placements) == 1
                assert not placements[0]["unsafe_reasons"], placements[0][
                    "unsafe_reasons"
                ]
                bitmap, mask, report = composite(
                    placements, images, [0, 0, 400, 200], pitch=1
                )
                samples = np.asarray(bitmap)
                assert all(samples[100, 50] == [255, 255, 255])
                assert all(samples[100, 100] == [0, 125, 215])
                assert all(samples[100, 335] == [255, 255, 255])
                assert report["paint_statistics"]["painted_instances"] == 1
                layouts.append(np.asarray(bitmap).copy())
            assert np.array_equal(layouts[0], layouts[1])


def test_device_raster_fast_decode_matches_full_pdf_decoder():
    """Uncompressed/Flate device pixels and complex fallback are bit-identical."""
    width, height = 4, 3
    rgb = bytes(range(width * height * 3))
    gray = bytes(range(width * height))
    cases = (
        ("raw_rgb", rgb, "/DeviceRGB", 8, False, None, False, False),
        ("flate_rgb", rgb, "/DeviceRGB", 8, True, None, False, False),
        ("raw_gray", gray, "/DeviceGray", 8, False, None, False, False),
        ("flate_gray", gray, "/DeviceGray", 8, True, None, False, False),
        (
            "rgb_explicit_identity",
            rgb,
            "/DeviceRGB",
            8,
            False,
            [0, 1, 0, 1, 0, 1],
            False,
            False,
        ),
        ("rgb_inverted", rgb, "/DeviceRGB", 8, False, [1, 0, 1, 0, 1, 0], False, False),
        ("gray_inverted", gray, "/DeviceGray", 8, True, [1, 0], False, False),
        ("rgb_soft_mask", rgb, "/DeviceRGB", 8, False, None, True, False),
        ("rgb_decode_params", rgb, "/DeviceRGB", 8, True, None, False, True),
        (
            "gray_1bit",
            bytes([0xA5, 0x3C, 0xC3]),
            "/DeviceGray",
            1,
            False,
            None,
            False,
            False,
        ),
    )
    with pikepdf.Pdf.new() as doc:
        for name, data, colorspace, bits, flate, decode, mask, params in cases:
            image = pikepdf.Stream(doc, zlib.compress(data) if flate else data)
            image["/Type"] = pikepdf.Name("/XObject")
            image["/Subtype"] = pikepdf.Name("/Image")
            image["/Width"] = width if bits == 8 else 8
            image["/Height"] = height
            image["/BitsPerComponent"] = bits
            image["/ColorSpace"] = pikepdf.Name(colorspace)
            if flate:
                image["/Filter"] = pikepdf.Name("/FlateDecode")
            if decode is not None:
                image["/Decode"] = pikepdf.Array(decode)
            if params:
                image["/DecodeParms"] = pikepdf.Dictionary(Predictor=1)
            if mask:
                alpha = pikepdf.Stream(doc, gray)
                alpha["/Type"] = pikepdf.Name("/XObject")
                alpha["/Subtype"] = pikepdf.Name("/Image")
                alpha["/Width"] = width
                alpha["/Height"] = height
                alpha["/BitsPerComponent"] = 8
                alpha["/ColorSpace"] = pikepdf.Name("/DeviceGray")
                image["/SMask"] = doc.make_indirect(alpha)
            with pikepdf.PdfImage(image).as_pil_image(
                apply_decode_array=True, apply_mask=True
            ) as reference:
                expected = np.asarray(reference.convert("RGBA"), dtype=np.uint8)
            fast = name in (
                "raw_rgb",
                "flate_rgb",
                "raw_gray",
                "flate_gray",
                "rgb_explicit_identity",
            )
            if fast:
                # Prove the eligible path does not instantiate PdfImage.
                with patch(
                    "reconstruct_raster_mosaic.pikepdf.PdfImage",
                    side_effect=AssertionError("unexpected full decoder"),
                ):
                    actual = decode_mosaic_image(image)
            else:
                actual = decode_mosaic_image(image)
            assert np.array_equal(actual, expected), name


def test_composite_fast_path_preserves_pixels_and_coverage():
    def original_decoder(image):
        with pikepdf.PdfImage(image).as_pil_image(
            apply_decode_array=True, apply_mask=True
        ) as raw:
            return np.asarray(raw.convert("RGBA"), dtype=np.uint8).copy()

    with TemporaryDirectory(prefix="pdf-redox-composite-fast-pixels-") as td:
        path = Path(td) / "fixture.pdf"
        make_fixture(path)
        with pikepdf.Pdf.open(path) as pdf:
            for page in (1, 2):
                paints, images, _ = enumerate_image_paints(pdf, [page])
                with patch(
                    "reconstruct_raster_mosaic.decode_mosaic_image",
                    side_effect=original_decoder,
                ):
                    expected_image, expected_coverage, expected_report = composite(
                        paints, images, [0, 0, 400, 200], pitch=1
                    )
                image, coverage, report = composite(
                    paints, images, [0, 0, 400, 200], pitch=1
                )
                assert np.array_equal(np.asarray(image), np.asarray(expected_image))
                assert np.array_equal(
                    np.asarray(coverage), np.asarray(expected_coverage)
                )
                assert report == expected_report


def _solid_rgb_xobject(doc, color, alpha=None):
    data = bytes(color) * (20 * 20)
    image = pikepdf.Stream(doc, data)
    image["/Type"] = pikepdf.Name("/XObject")
    image["/Subtype"] = pikepdf.Name("/Image")
    image["/Width"] = 20
    image["/Height"] = 20
    image["/BitsPerComponent"] = 8
    image["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
    if alpha is not None:
        mask = pikepdf.Stream(doc, bytes([alpha]) * (20 * 20))
        mask["/Type"] = pikepdf.Name("/XObject")
        mask["/Subtype"] = pikepdf.Name("/Image")
        mask["/Width"] = 20
        mask["/Height"] = 20
        mask["/BitsPerComponent"] = 8
        mask["/ColorSpace"] = pikepdf.Name("/DeviceGray")
        image["/SMask"] = doc.make_indirect(mask)
    return doc.make_indirect(image)


def test_premultiplied_raster_edges_and_masked_layers():
    """Opaque blue must remain blue under fractional-edge antialiasing."""
    with pikepdf.Pdf.new() as doc:
        blue = _solid_rgb_xobject(doc, (0, 0, 255))
        paint = {
            "source_id": "blue",
            "width": 20,
            "height": 20,
            "ctm": [27, 0, 0, 27, 10.25, 12.25],
            "unsafe_reasons": [],
            "clip_polygons": [],
        }
        bitmap, coverage, report = composite(
            [paint], {"blue": blue}, [0, 0, 50, 50], pitch=0.5
        )
        pixels = np.asarray(bitmap)
        # With straight-alpha interpolation this fixture produced 647 dark
        # pixels and the blue channel dropped all the way to 191.
        assert pixels[:, :, 2].min() >= 254
        assert np.count_nonzero(np.asarray(coverage)) > 2500
        assert report["paint_statistics"]["painted_instances"] == 1

        blue_masked = _solid_rgb_xobject(doc, (0, 0, 255), alpha=128)
        red_masked = _solid_rgb_xobject(doc, (255, 0, 0), alpha=128)
        blue_paint = {**paint, "source_id": "blue_masked"}
        red_paint = {
            **paint,
            "source_id": "red_masked",
            "clip_polygons": [[[10, 10], [20, 10], [20, 50], [10, 50]]],
        }
        image, _, report = composite(
            [blue_paint, red_paint],
            {"blue_masked": blue_masked, "red_masked": red_masked},
            [0, 0, 50, 50],
            pitch=0.5,
        )
        data = np.asarray(image)
        assert report["paint_statistics"]["painted_instances"] == 2
        # The blue source over white produces (127,127,255). The red
        # semitransparent paint then covers only its clipping rectangle.
        red_over_blue = data[60, 30].astype(np.int16)
        blue_only = data[60, 50].astype(np.int16)
        assert np.max(np.abs(red_over_blue - [191, 63, 127])) <= 2, red_over_blue
        assert np.max(np.abs(blue_only - [127, 127, 255])) <= 2, blue_only


def reference_quadratic_groups(placements, page_rect, proximity=2.5, tolerance=0.075):
    """Oracle: the original all-pairs grouping rule, without spatial indexing."""
    items = []
    for p in placements:
        if p["unsafe_reasons"]:
            continue
        w, h = p["width"], p["height"]
        if w <= 0 or h <= 0:
            continue
        c = p["ctm"]
        pitch_x = math.hypot(c[0], c[1]) / w
        pitch_y = math.hypot(c[2], c[3]) / h
        pitch = (pitch_x + pitch_y) / 2
        if not 0.04 < pitch <= 10 or abs(pitch_x - pitch_y) > tolerance * pitch:
            continue
        corners = image_quad(p, [[0, 0], [w, 0], [w, h], [0, h]])
        box = (
            max(page_rect[0], min(x for x, _ in corners)),
            max(page_rect[1], min(y for _, y in corners)),
            min(page_rect[2], max(x for x, _ in corners)),
            min(page_rect[3], max(y for _, y in corners)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        items.append((p, box, pitch))
    parents = list(range(len(items)))

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, (_, a, pitch_a) in enumerate(items):
        for j in range(i + 1, len(items)):
            _, b, pitch_b = items[j]
            if abs(pitch_a - pitch_b) > tolerance * max(pitch_a, pitch_b):
                continue
            xgap = max(0, a[0] - b[2], b[0] - a[2])
            ygap = max(0, a[1] - b[3], b[1] - a[3])
            gap_limit = proximity * max(pitch_a, pitch_b)
            if xgap <= gap_limit and ygap <= gap_limit:
                parents[root(j)] = root(i)
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(root(i), []).append(item)
    result = []
    for group in groups.values():
        if len(group) < 2:
            continue
        pitch = statistics.median(z[2] for z in group)
        region = (
            max(page_rect[0], min(z[1][0] for z in group) - 5 * pitch),
            max(page_rect[1], min(z[1][1] for z in group) - 5 * pitch),
            min(page_rect[2], max(z[1][2] for z in group) + 5 * pitch),
            min(page_rect[3], max(z[1][3] for z in group) + 5 * pitch),
        )
        pixels = math.ceil((region[2] - region[0]) / pitch) * math.ceil(
            (region[3] - region[1]) / pitch
        )
        if pixels <= 12_000_000:
            result.append(
                (frozenset(p["source_id"] for p, _, _ in group), region, pitch, pixels)
            )
    return sorted(result, key=lambda x: sorted(x[0]))


def test_spatial_index_agrees_with_all_pairs():
    rng = random.Random(314159)
    page = (-50, -50, 850, 650)
    for trial in range(20):
        placements = []
        for i in range(180):
            pitch = rng.choice((0.6, 0.62, 0.65, 0.75, 1.0, 1.05, 1.1, 1.3))
            w, h = rng.choice((12, 24, 60, 180)), rng.choice((6, 12, 40, 200))
            x = rng.uniform(-90, 900) if trial % 2 else (i % 15) * 18
            y = rng.uniform(-90, 700) if trial % 2 else (i // 15) * 16
            transform = (
                [0, w * pitch, -h * pitch, 0, x, y]
                if i % 11 == 0
                else [w * pitch, 0, 0, h * pitch, x, y]
            )
            placements.append(
                {
                    "source_id": f"source-{i}",
                    "width": w,
                    "height": h,
                    "ctm": transform,
                    "unsafe_reasons": ["unsupported"] if i % 23 == 0 else [],
                }
            )
        expected = reference_quadratic_groups(placements, page)
        groups = cluster_native_raster_paints(placements, page)
        found = sorted(
            [
                (
                    frozenset(p["source_id"] for p in group["placements"]),
                    group["region"],
                    group["pitch"],
                    group["estimated_pixels"],
                )
                for group in groups
            ],
            key=lambda x: sorted(x[0]),
        )
        assert found == expected, (trial, found, expected)
        reordered = rng.sample(placements, k=len(placements))
        permuted = sorted(
            [
                (
                    frozenset(p["source_id"] for p in group["placements"]),
                    group["region"],
                    group["pitch"],
                    group["estimated_pixels"],
                )
                for group in cluster_native_raster_paints(reordered, page)
            ],
            key=lambda x: sorted(x[0]),
        )
        assert permuted == expected, (trial, permuted, expected)


def test_spatial_index_large_footprint_and_symmetric_pitch():
    page = (0, 0, 900, 900)
    # One image spans more than the bucket budget; later small paints must
    # still find it, regardless of index insertion or query order.
    large = {
        "source_id": "large",
        "width": 600,
        "height": 600,
        "ctm": [600, 0, 0, 600, 20, 20],
        "unsafe_reasons": [],
    }
    small = {
        "source_id": "small",
        "width": 10,
        "height": 10,
        "ctm": [10, 0, 0, 10, 300, 300],
        "unsafe_reasons": [],
    }
    distant = {
        "source_id": "distant",
        "width": 10,
        "height": 10,
        "ctm": [10, 0, 0, 10, 720, 720],
        "unsafe_reasons": [],
    }
    for ordered in ([large, small, distant], [small, distant, large]):
        result = cluster_native_raster_paints(ordered, page)
        assert len(result) == 1, result
        assert {p["source_id"] for p in result[0]["placements"]} == {"large", "small"}

    def strip(source, x, pitch):
        return {
            "source_id": source,
            "width": 10,
            "height": 10,
            "ctm": [10 * pitch, 0, 0, 10 * pitch, x, 20],
            "unsafe_reasons": [],
        }

    # Previously a 2.6 pt gap grouped only if the 1.05-pitch source
    # appeared first. The correct symmetric rule accepts either order.
    first = strip("first", 20, 1)
    second = strip("second", 32.6, 1.05)
    assert len(cluster_native_raster_paints([first, second], page)) == 1
    assert len(cluster_native_raster_paints([second, first], page)) == 1


if __name__ == "__main__":
    test_premultiplied_raster_edges_and_masked_layers()
    test_device_raster_fast_decode_matches_full_pdf_decoder()
    test_composite_fast_path_preserves_pixels_and_coverage()
    test_spatial_index_agrees_with_all_pairs()
    test_spatial_index_large_footprint_and_symmetric_pitch()
    test_clips_and_mosaic()
    print("PDF_NATIVE_RASTER_MOSAIC_CLIPPING_PASS")
