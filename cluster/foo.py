from PIL import Image
 
# Opens a image in RGB mode
im = Image.open('./example_image.jpeg').resize((128,128))
im.save('example3.jpeg')

