import time
import cv2
import argparse
from PIL import Image
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

current_upscale = ""
failed_images = {}


def convert_to_png(image_path):
    """
    Converts an image to PNG format if it is not already in PNG format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Path to the converted PNG image file.
    """
    start_name = image_path[:]
    if not image_path.endswith("png"):
        image = Image.open(image_path)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        file_name = os.path.splitext(image_path)[0]
        output_path = f"{file_name}.png"
        image.save(output_path, "PNG")
        image.close()
        os.remove(start_name)
        print(f"Converted {start_name} to png")
        return output_path
    else:
        return image_path


def list_folders(pth):
    """
        Returns a list of folder names within the specified path.

        Args:
            pth (str): Path to the directory.

        Returns:
            list: List of folder names.
    """
    folders = []
    for root, dir_names, _ in os.walk(pth):
        for dir_name in dir_names:
            folders.append(dir_name)
    return folders


def files_in_folder(folder_path):
    """
        Retrieves a list of image files within the specified folder.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            list: List of image file paths.
    """
    image_extensions = ['.jpg', '.jpeg', '.png']  # Add more extensions if needed
    image_files = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if the file has a supported image extension
        if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)

    return image_files


def upscale(model_path, im_path):
    """
       Upscales an image using the specified model.

       Args:
           model_path (str): Path to the model being used.
           im_path (str): Path to the input image file.

       Returns:
           ndarray or None: Upscaled image as NumPy array or None if there was an error.
    """
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    up_sampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    try:
        output_image, _ = up_sampler.enhance(img, outscale=4)
        return output_image
    except Exception as e:
        print(f"\nError in up scaling {current_upscale}\n{e}")
        return None


def upscale_image(input_image_path, output_image_path):
    """
        Upscales an image and saves the output to the specified path.

        Args:
            input_image_path (str): Path to the input image file.
            output_image_path (str): Path to save the upscaled image.

        Returns:
            None
    """
    start_time = time.perf_counter()
    global MODEL_PATH
    global current_upscale
    args.model_path = MODEL_PATH
    args.input = convert_to_png(input_image_path)
    args.output = output_image_path
    print(f"up scaling {args.input}", end=" ")
    if not args.input == current_upscale:
        if args.model_path and args.input and args.output:
            output = upscale(args.model_path, args.input)
            if output is None:
                return
            cv2.imwrite(args.output, output)
        else:
            print('Error: Missing arguments, check -h, --help for details')
        current_upscale = args.input
        print(f"{round((time.perf_counter() - start_time) / 6 ) / 10} min")


def upscale_folder(input_folder_path, output_folder_path):
    """
        Upscales all images in a folder and saves the output to the specified folder.

        Args:
            input_folder_path (str): Path to the input folder.
            output_folder_path (str): Path to save the upscaled images.

        Returns:
            None
    """
    if output_folder_path == "":
        output_folder_path = os.path.join(input_folder_path, "UPSCALED")
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
    if os.path.exists(input_folder_path) and os.path.exists(output_folder_path):
        start_time = time.perf_counter()
        print(f">>UP SCALING {input_folder_path}\n")
        for image in files_in_folder(input_folder_path):
            upscale_image(image, os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(image))[0]}.png"))
        print(f"\n{input_folder_path} IS UP SCALED; Time taken: {round((time.perf_counter() - start_time) / 6 )/10} min\n\n\n")
    else:
        print(f"Input/Output folder doesnt exists.")


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


def invalid_input(inp):
    """
    Prints an error message for invalid input and exits the program.

    Args:
        inp (str): Invalid input.

    Returns:
        None
    """
    print(f"Invalid {inp}")
    exit()


if __name__ == '__main__':
    # INPUTS FROM USER
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type_of_upscale', type=str, required=True, help='REQUIRED: specify weather up scaling image ot a folder')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='REQUIRED: specify path of the model being used')
    parser.add_argument('-i', '--input', type=str, required=True, help='REQUIRED: specify path of the input file/image')
    parser.add_argument('-o', '--output', type=str, help='REQUIRED: specify path of the output (optional) file/image')
    parser.add_argument('-c', '--colour_upscale', type=bool, help='REQUIRED: specify weather to colour upscale or not')
    args = parser.parse_args()

    # CHECKING IF THE INPUTS ARE CORRECT
    invalid_input("argument for type_of_upscale") if args.type_of_upscale != "1" and args.type_of_upscale != "2" else None
    invalid_input("model_path") if not os.path.exists(args.model_path) else None
    MODEL_PATH = args.model_path
    invalid_input("input_path") if not os.path.exists(args.input) else None
    if args.output is None:
        # MAKING AN OUTPUT PATH
        directory, filename = os.path.split(args.input)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_upscaled{extension}'
        args.output = os.path.join(directory, new_filename)
        os.mkdir(args.output)
########################################################################################################################
    # INFORMING THE FUNCTIONS TO UPSCALE
    upscale_image(args.input, args.output) if args.type_of_upscale == "1" else upscale_folder(args.input, args.output)
    """if args.colour_upscale:
        pass
    else:
        upscale_image(args.input, args.output) if args.type_of_upscale == "1" else upscale_folder(args.input, args.output)
        """
    

