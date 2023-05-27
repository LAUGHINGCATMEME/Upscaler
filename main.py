import time
import cv2
import argparse
import image_slicer
from image_slicer import join
from PIL import Image
import numpy as np
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

current_upscale = ""


def convert_to_png(image_path):
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
    folders = []
    for root, dir_names, _ in os.walk(pth):
        for dir_name in dir_names:
            folders.append(dir_name)
    return folders


def get_image_files(folder_path):
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
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    up_sampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    try:
        output_image, _ = up_sampler.enhance(img, outscale=4)
        return output_image
    except Exception as e:
        print(f"\nError in up scaling {current_upscale}\n{e}")
        return None


def upscale_slice(model_path, image, slicer):
    width, height = Image.open(image).size
    tiles = image_slicer.slice(image, slicer, save=False)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    up_sampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
    for tile in tiles:
        output_image, _ = up_sampler.enhance(np.array(tile.image), outscale=4)
        tile.image = Image.fromarray(output_image)
        tile.coords = (tile.coords[0] * 4, tile.coords[1] * 4)
    return join(tiles, width=width * 4, height=height * 4)


def upscale_image(input_image_path, output_image_path):
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
    if output_folder_path == "":
        output_folder_path = os.path.join(input_folder_path, "UPSCALED")
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
    if os.path.exists(input_folder_path) and os.path.exists(output_folder_path):
        start_time = time.perf_counter()
        print(f">>UP SCALING {input_folder_path}\n")
        for image in get_image_files(input_folder_path):
            upscale_image(image, os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(image))[0]}.png"))
        print(f"\n{input_folder_path} IS UP SCALED; Time taken: {round((time.perf_counter() - start_time) / 6 )/10} min\n\n\n")
    else:
        print(f"Input/Output folder doesnt exists.")


def invalid_input(inp):
    print(f"Invalid {inp}")
    exit()


if __name__ == '__main__':
    # INPUTS FROM USER
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type_of_upscale', type=str, required=True, help='REQUIRED: specify weather up scaling image ot a folder')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='REQUIRED: specify path of the model being used')
    parser.add_argument('-i', '--input', type=str, required=True, help='REQUIRED: specify path of the input')
    parser.add_argument('-o', '--output', type=str, help='REQUIRED: specify path of the output (optional)')
    args = parser.parse_args()

    # CHECKING IF THE INPUTS ARE CORRECT
    invalid_input("argument for type_of_upscale") if args.type_of_upscale != "1" and args.type_of_upscale != "2" else None
    invalid_input("model_path") if not os.path.exists(args.model_path) else None
    MODEL_PATH = args.model_path
    invalid_input("input_path") if not os.path.exists(args.input) else None
    if args.output is None:
        # Original file path
        directory, filename = os.path.split(args.input)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_upscaled{extension}'
        args.output = os.path.join(directory, new_filename)
    else:
        invalid_input("output_path") if not os.path.exists(args.output) else None

########################################################################################################################
    if args.type_of_upscale == "1":
        upscale_image(args.input, args.output)
    exit()
