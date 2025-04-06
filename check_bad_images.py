
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# Set your dataset folder path here
folder = Path(r"C:\path\to\your\images")  # 👈 replace with your actual path

# Collect all .jpg and .jpeg files, recursively
image_files = list(folder.rglob("*.jpg")) + list(folder.rglob("*.jpeg"))

bad_images = []

for img_path in image_files:
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify it's a valid image
    except (UnidentifiedImageError, OSError) as e:
        bad_images.append(str(img_path))
        print(f"❌ Bad image: {img_path}")

print(f"\n✅ Done scanning. Found {len(bad_images)} bad image(s).")
