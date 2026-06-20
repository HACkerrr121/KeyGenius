from PIL import Image, ImageFilter
img = Image.open("score.jpg").convert("L")            # grayscale
img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80))  # gentle sharpen
w, h = img.size
img = img.resize((int(w*1.8), int(h*1.8)), Image.LANCZOS)  # upscale
img.save("score_hq.png", dpi=(300, 300))               # tag 300 DPI