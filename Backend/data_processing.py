from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

'''folder = Path("Music_Data/Scores")
for item in folder.iterdir():
    if item.suffix.lower() == '.jpg':  # Only process PDF files
        images = convert_from_path(item)  # Use the actual file path
        for i, image in enumerate(images):
            print(image)
            #image.save(f'{item.stem}_page_{i}.jpg', 'JPEG')  # Include filename in output'''

image  = Image.open("Music_Data/Scores/038_Debussy_LaFilleAuxCheveuxDeLin_B1-10_page_0.jpg")

width, height = image.size
print(width, height)