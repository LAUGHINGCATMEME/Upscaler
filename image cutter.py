from PIL import Image


def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except IOError:
        print("Unable to open image file:", image_path)
        return None


def split_image(image_path, original_width, original_height, part_width, part_height):
    try:
        with Image.open(image_path) as img:
            for i in range(int(original_width / part_width)):
                for j in range(int(original_height / part_height)):
                    left = j * part_width
                    upper = i * part_height
                    right = left + part_width
                    lower = upper + part_height

                    part_img = img.crop((left, upper, right, lower))
                    part_img.save(f"C2\\part_{i}_{j}.png")  # Change the filename and extension as needed

            print(f"Image split into {int(original_width / part_width) * int(original_height / part_height)} parts successfully.")
    except IOError:
        print("Unable to open image file:", image_path)


def combine_images(original_width, original_height, part_width, part_height):
    combined_width = original_width
    combined_height = original_height
    combined_image = Image.new("RGB", (combined_width, combined_height))

    for i in range(int(original_width / part_width)):
        for j in range(int(original_height / part_height)):
            part_filename = f"part_{i}_{j}.png"  # Adjust the filename format if needed
            part_image = Image.open(part_filename)
            combined_image.paste(part_image, (j * part_width, i * part_height))

    combined_image.save("BROOOOOOOOOO.png")  # Change the filename and extension as needed
    print("Images combined into one.")


# Example usage
image_path = "C:\\Users\\Lenovo\\Desktop\\la2.png"
original_width = 25600
original_height = 19200
part_width = 3200
part_height = 2400

#split_image(image_path, original_width, original_height, part_width, part_height)
combine_images(original_width, original_height, part_width, part_height)
