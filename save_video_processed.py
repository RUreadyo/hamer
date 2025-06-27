import cv2
import os


# Set the directory containing your images
task_names = [
    "closelaptop_prev",
    "closemicrowave",
    "dragbasket",
    "ironing",
    "pickplacebanana",
    "pickplacebowl",
    "wiping",
    "pickupbear"
]
traj_list = [0,1]
cam_ids = [1, 2, 3, 4]

for task_name in task_names:
    for traj in traj_list:
        for cam_id in cam_ids:

            image_folder = f"./{task_name}/traj_{traj}/{cam_id}/processed"
            output_video = f"./{task_name}/traj_{traj}/output_{cam_id}.mp4"

            fps = 15  # Set your desired frame rate

            # Get sorted list of image filenames
            images = [
                img for img in os.listdir(image_folder) if img.endswith("all.jpg")
            ]
            images.sort()  # Ensures correct order

            # Read the first image to get frame size
            first_image = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = first_image.shape
            size = (width, height)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )  # 'mp4v' is commonly used for .mp4 files
            out = cv2.VideoWriter(output_video, fourcc, fps, size)

            for image in images:
                img_path = os.path.join(image_folder, image)
                frame = cv2.imread(img_path)
                out.write(frame)

            out.release()
            print("Video created successfully!")
