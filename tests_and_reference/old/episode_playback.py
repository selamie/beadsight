# Plays back the recorded information stored in an hdf5 file.
import argparse
import h5py
from typing import List, Tuple
import cv2
import numpy as np

def visualize_gelsight_data(image):
	# Convert the image to LAB color space
	max_depth = 10
	max_strain = 30
	# Show all three using LAB color space
	image[0] = np.clip(100*np.maximum(image[0], 0)/max_depth, 0, 100)
	# normalized_depth = np.clip(100*(depth_image/depth_image.max()), 0, 100)
	image[1:] = np.clip(128*(image[1:]/max_strain), -128, 127)
	return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

def main(args: argparse.Namespace):
    file_path = args.file_path

    with h5py.File(file_path, 'r') as root:
        attrs = dict(root.attrs)
        print("run attributes")
        print("-"*30)
        for key, value in attrs.items():
            print(f"{key}: {value}")

        position = root['/observations/position']
        velocity = root['/observations/velocity']
        goal_position = root['/goal_position']
        images = root['/observations/images/']
        image_attrs = dict(images.attrs)

        # check if the image_attrs has a compression key
        if 'compression' in image_attrs:
            compression = image_attrs['compression']
        else:
            compression = None

        if attrs['use_gelsight']:
            gelsight_raw_images = root['/observations/gelsight/raw_image']
            gelsight_depth_strain_images = root['/observations/gelsight/depth_strain_image']

        for i in range(attrs['num_timesteps']):
            print('\nTimestep: ', i)
            print('Position: ', position[i, :])
            print('Velocity: ', velocity[i, :])
            print('Goal Position: ', goal_position[i, :])

            if compression is None:
                for camera in attrs['camera_names'].astype(str):
                    cv2.imshow(camera, images[camera][i, :, :, :])
            elif compression == 'jpeg':
                for camera in attrs['camera_names'].astype(str):
                    cv2.imshow(camera, cv2.imdecode(images[camera][i], cv2.IMREAD_COLOR))
            else:
                raise ValueError(f"Unknown compression type: {compression}")
            
            cv2.imshow("gelsight raw image", gelsight_raw_images[i, :, :, : ])
            # convert the depth strain image to lab color space
            cv2.imshow("gelsight depth strain image", visualize_gelsight_data(gelsight_depth_strain_images[i, :, :, :]))
            cv2.waitKey(50)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, default=None, help='Path to the hdf5 file to be played back.')
    args = parser.parse_args()
    main(args)