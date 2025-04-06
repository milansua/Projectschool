
from pathlib import Path

# Set your target folder path
folder = Path(r"C:\path\to\your\images")  # 👈 Replace with your folder path

# Supported image extensions
extensions = ['.jpg', '.jpeg', '.png', '.webp']

# Rename files
for img in folder.iterdir():
    if img.is_file() and img.suffix.lower() in extensions and "_" in img.name:
        new_name = img.name.replace("_", "-")
        new_path = folder / new_name
        img.rename(new_path)
        print(f"Renamed: {img.name} -> {new_name}")

print("✅ Done renaming files.")
