import base64
import io

from PIL import Image

from polehealth import images


def test_resize_never_upscales_and_keeps_aspect():
    small = Image.new("RGB", (400, 300))
    assert images.resize_max_edge(small, 1024) is small
    big = Image.new("RGB", (4000, 3000))
    assert images.resize_max_edge(big, 1024).size == (1024, 768)


def test_exif_orientation_is_applied(tmp_path):
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    path = tmp_path / "rot.jpg"
    image.save(path, "JPEG", exif=exif)
    assert images.load_image(path).size == (100, 200)


def test_data_url_is_jpeg_within_the_model_limits(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), (10, 20, 30)).save(path)
    url = images.data_url(path)
    header, payload = url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    with Image.open(io.BytesIO(base64.b64decode(payload))) as decoded:
        assert decoded.format == "JPEG" and max(decoded.size) == 1024


def test_demo_photo_is_max_960_jpeg(tmp_path):
    src = tmp_path / "in.jpg"
    Image.new("RGB", (3000, 4000), (10, 20, 30)).save(src)
    dst = tmp_path / "out" / "p.jpg"
    assert images.save_demo_photo(src, dst) == (720, 960)
    with Image.open(dst) as out:
        assert out.format == "JPEG" and out.size == (720, 960)
