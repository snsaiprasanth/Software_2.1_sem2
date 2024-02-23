import cv2
import os

def convert_images_to_hsv(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is not None:
            # Convert the image to HSV
            hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

            # Save the HSV image to the output folder
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, hsv_image)
            print(f"Image {image_file} converted and saved as {output_path}")

def convert_images_to_gray(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is not None:
            # Convert the image to gray
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Save the gar image to the output folder
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, gray_image)
            print(f"Image {image_file} converted and saved as {output_path}")

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = r"D:\2024_WORKSHOP_IAAC\3D_model\input\iaac_building_02"
    output_folder = r"D:\2024_WORKSHOP_IAAC\3D_model\input\iaac_building_gray"

    # Call the function to convert images to HSV
    # convert_images_to_hsv(input_folder, output_folder)
    convert_images_to_gray(input_folder, output_folder)
