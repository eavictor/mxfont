import argparse
import os
import sys
try:
    import fontforge
except ModuleNotFoundError:
    print("""
    ffpython is required to run this script !!!
    
    1. Download (Click on your platform name): https://fontforge.org/en-US/downloads/
    2. Install FontForge.
    3. Add ffpython to path (require reboot if using windows), or call ffpython directly with full path.
    4. Execute this script with ffpython.
    
    Command Example: ffpython svg2ttf <svg_folder> <output_font_name>
    """)


def create_new_font(svg_folder: str, font_name: str):
    try:
        font = fontforge.font()
        font.encoding = "UnicodeFull"
        if os.path.isdir(svg_folder):
            characterAndOutlineFilename = {}
            svg_files = os.listdir(svg_folder)
            for svg_file in svg_files:
                char_order = svg_file.replace(".svg", "")
                characterAndOutlineFilename[char_order] = os.path.join(svg_folder, svg_file)
            for char_order, svg_file in characterAndOutlineFilename.items():
                glyph = font.createChar(char_order)
                glyph.importOutlines(svg_file)
            if font_name.endswith(".ttf"):
                font.save(font_name)
            else:
                font.save(f"{font_name}.ttf")
    except NameError:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_folder", type=str, help="Folder containing SVG files")
    parser.add_argument("font_name", type=str, help="Name of font to use")
    args = parser.parse_args()
    create_new_font(args.svg_folder, args.font_name)
