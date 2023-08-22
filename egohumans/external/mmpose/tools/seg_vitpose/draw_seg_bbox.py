import cv2
import numpy as np
import argparse
import os

def draw_mask(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['drawing'] = True
        param['current_contour'].append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['drawing'] == True:
            cv2.circle(param['mask'], (x, y), 5, 255, -1)
            param['current_contour'].append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        param['drawing'] = False
        cv2.circle(param['mask'], (x, y), 5, 255, -1)
        param['current_contour'].append((x, y))
        param['contours'].append(param['current_contour'])
        param['current_contour'] = []

def create_colored_mask(mask):
    blue_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)
    blue_mask[mask == 255] = (255, 0, 0)  # Set the mask color to blue (BGR)
    return blue_mask

def add_glow_effect(mask, intensity=10, kernel_size=21):
    blur_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    glow_mask = cv2.addWeighted(mask, 1, blur_mask, intensity, 0)
    return glow_mask

def main(image_path, mask_path, overlay_path):
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Image not found.')
        return

    height, width = image.shape[:2]
    fixed_height = 720
    fixed_width = int(width * (fixed_height / height))
    resized_image = cv2.resize(image, (fixed_width, fixed_height))

    mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)

    cv2.namedWindow('Image with Mask')
    param = {
        'drawing': False,
        'current_contour': [],
        'contours': [],
        'mask': mask
    }
    cv2.setMouseCallback('Image with Mask', draw_mask, param)

    while True:
        display_image = cv2.addWeighted(resized_image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cv2.imshow('Image with Mask', display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    filled_mask = np.zeros_like(mask)
    for contour in param['contours']:
        cv2.fillPoly(filled_mask, np.array([contour], dtype=np.int32), 255)

    original_size_mask = cv2.resize(filled_mask, (width, height))
    colored_mask = create_colored_mask(original_size_mask)
    glow_mask = add_glow_effect(colored_mask)

    overlay_image = cv2.addWeighted(image, 0.7, glow_mask, 0.3, 0)

    mask_dir = os.path.dirname(mask_path)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    overlay_dir = os.path.dirname(overlay_path)
    if not os.path.exists(overlay_dir):
        os.makedirs(overlay_dir)

    cv2.imwrite(mask_path, original_size_mask)
    print(f'Saved binary mask as a black and white image: {mask_path}')

    cv2.imwrite(overlay_path, overlay_image)
    print(f'Saved overlay image: {overlay_path}')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw a mask on an image and save it.')
    parser.add_argument('image_path', help='Path to the input image.')
    parser.add_argument('mask_path', help='Path to save the binary mask.')
    parser.add_argument('overlay_path', help='Path to save the overlay image.')
    args = parser.parse_args()

    main(args.image_path, args.mask_path, args.overlay_path)

