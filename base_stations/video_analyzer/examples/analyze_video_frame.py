import json
import os

import requests


def process_image(image_path, server_url="http://localhost:5000/process"):
    """Send an image to the video frame analyzer server and get the results.

    Args:
        image_path (str): Path to the image file.
        server_url (str): URL of the video frame analyzer server.

    Returns:
        dict: Processed results from the server.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, 'rb') as image_file:
        files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
        response = requests.post(server_url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error processing image: {response.text}")


def main():
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    image_path = os.path.join(project_dir, '/Users/ericli/Desktop/Conference/IMWUT/experiments/session_2025-01-22T14'
                                           ':42:37Z/video_frames/frame_1737558012.jpg')
    try:
        results = process_image(image_path)
        print("Image processing results:")
        print(json.dumps(results, indent=2))

        image_description = results['image_description']
        categorization_result = results['categorization_result']

        print("\nImage Description:")
        print(image_description)

        print("\nCategorization Result:")
        # for category, description in categorization_result.items():
        #     print(f"{category}: {description}")

        for pair in categorization_result['pairs']:
            print(f"ID: {pair['id']}, Action: {pair['action']}")


    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
