import cv2
import numpy as np
from scipy.ndimage import map_coordinates

class Camera:
    def __init__(self, cam_id=0, output_shape=None):
        self.cam_id = cam_id
        self.output_shape = output_shape or (1280, 800)
        self.set_capture_device()
        self.read_image_properties()

    def set_capture_device(self):
        self.camera = cv2.VideoCapture(self.cam_id)

    def read_capture_device(self):
        ack, img = self.camera.read()
        if not ack:
            raise ValueError('Image could not be read from device')        
        return img

    def read_image_properties(self):
        img = self.read_capture_device()
        self.shape = img.shape[:2]
    
    def show(self, image=None):
        image = image if image is not None else self.read_capture_device()
        image = cv2.resize(image, self.output_shape)
        cv2.imshow('Face Lensing', image)

    def switch_capture_device(self):
        self.cam_id = 1 - self.cam_id   
        self.release()
        print(f'Switching to camera {self.cam_id}')
        self.set_capture_device()
        self.read_image_properties()
    
    def release(self):
        self.camera.release()


class Morphing:
    def __init__(self, target_shape, morph_file, zoom, shift=(0, 0)):
        self.zoom = zoom
        self.shift = np.asarray(shift, dtype=int)
        self.read_morph_file(morph_file)
        self.set_target_shape(target_shape)

    def read_morph_file(self, file_path):
        lens_model = np.load(file_path)
        self.hx = lens_model["dplx"]
        self.hy = lens_model["dply"]
        self.shape = np.asarray(self.hx.shape, dtype=int)

    def set_target_shape(self, new_shape):
        self.target_shape = np.asarray(new_shape, dtype=int)
        self.init_morphing()

    def init_morphing(self):
        height, width = self.target_shape
        lens_center = self.shape / 2
        img_center = self.target_shape / 2 + self.shift

        self.Y, self.X = np.indices(self.target_shape)

        # Create coordinates mapping between the image and the lens
        centred_coords = (
            np.indices(self.target_shape).reshape(2, -1) 
            - img_center[:, None] 
            + lens_center[:, None]).astype(int)

        centred_coords[0] = centred_coords[0].clip(0, self.shape[0]-1)
        centred_coords[1] = centred_coords[1].clip(0, self.shape[1]-1)

        lens_coords = np.reshape(centred_coords, (2, *self.target_shape))

        dpl1y = map_coordinates(self.hy / self.zoom, lens_coords, order=1)
        dpl1x = map_coordinates(self.hx / self.zoom, lens_coords, order=1)

        self.dply = np.array(self.Y - dpl1y).astype(int)
        self.dplx = np.array(self.X - dpl1x).astype(int)
        # Mirror lens effect on the horizontal axis
        self.dplx = width - self.dplx

    def apply(self, image):
        return np.asarray(
            list(map(lambda p, q: image[self.dply[p, q], self.dplx[p, q]], self.Y, self.X)))


def main(lens_file="dpl_xy_z1_elliptical.npz", cam_id=0, zoom=0.07):
    cam = Camera(cam_id)
    morph = Morphing(cam.shape, lens_file, zoom)
    img_display = cam.read_capture_device()

    while True:
        img_display[...] = morph.apply(
            cam.read_capture_device()
        )
        cam.show(img_display)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            cam.switch_capture_device()
            morph.set_target_shape(cam.shape)
            img_display = cam.read_capture_device()

        if cv2.waitKey(1) & 0xFF == ord('j'):
            print("augmente effet")
            morph.zoom -= 0.005
            morph.init_morphing()

        if cv2.waitKey(1) & 0xFF == ord('k'):
            print("diminue effet")
            morph.zoom += 0.005
            morph.init_morphing()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    