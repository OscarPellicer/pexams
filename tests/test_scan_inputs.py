import pytest
from PIL import Image

from pexams.scan_reading import load_scan_pages


def _pdf(path, shades):
    """Write a PDF with one small page per grey shade, so pages can be told apart."""
    pages = [Image.new("RGB", (60, 80), (shade, shade, shade)) for shade in shades]
    pages[0].save(path, save_all=True, append_images=pages[1:])
    return str(path)


def _shades(images):
    return [int(img[40, 30, 0]) for img in images]


def test_several_pdfs_are_read_in_the_order_given(tmp_path):
    first = _pdf(tmp_path / "b_first.pdf", [10, 20])
    second = _pdf(tmp_path / "a_second.pdf", [30])

    images, sources, _ = load_scan_pages([first, second])

    assert _shades(images) == pytest.approx([10, 20, 30], abs=3)
    assert sources == [f"{first}#page=1", f"{first}#page=2", f"{second}#page=1"]


def test_folder_mixes_images_and_pdfs_sorted_by_name(tmp_path):
    _pdf(tmp_path / "2_scan.pdf", [40, 50])
    Image.new("RGB", (60, 80), (10, 10, 10)).save(tmp_path / "1_page.png")
    _pdf(tmp_path / "3_scan.pdf", [60])
    (tmp_path / "notes.txt").write_text("ignored")

    images, sources, _ = load_scan_pages(str(tmp_path))

    assert _shades(images) == pytest.approx([10, 40, 50, 60], abs=3)
    assert sources[0].endswith("1_page.png")


def test_single_pdf_path_still_works(tmp_path):
    images, _, _ = load_scan_pages(_pdf(tmp_path / "scan.pdf", [10, 20, 30]))

    assert len(images) == 3


def test_invalid_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a PDF file or a directory"):
        load_scan_pages([str(tmp_path / "missing.pdf")])
