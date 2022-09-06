import cv2
import numpy as np

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
    
    def show(self, image):
        image = cv2.resize(image, self.output_shape)
        cv2.imshow('video test', image)

    def switch_capture_device(self):
        self.cam_id = 1 - self.cam_id
        self.release()
        print(f'Switching to camera {self.cam_id}')
        self.set_capture_device()
        self.read_image_properties()
    
    def release(self):
        self.camera.release()


class Morphing:
    def __init__(self, target_shape, morph_file, zoom=0.07, shift=(0, 0)):
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
        mapcentery = self.shape[0] / 2
        mapcenterx = self.shape[1] / 2  # because both maps have the same size
        centery = height / 2 + self.shift[0]
        centerx = width / 2 + self.shift[1]

        dplx = np.empty([height, width], dtype=int)
        dply = np.empty([height, width], dtype=int)

        Y, X = np.meshgrid(np.arange(height), np.arange(width))

        Y = np.asarray(list(map(lambda p: mapcentery+(p-centery), Y)), dtype=int)
        Y.clip(0, self.shape[0]-1)
        X = np.asarray(list(map(lambda q: mapcenterx+(q-centerx), X)), dtype=int)
        X.clip(0, self.shape[1]-1)

        dplx[:, :] = np.asarray(
            list(map(lambda p, q: q-self.hx[Y[p, q], X[p, q]]/self.zoom, Y, X)))
        dplx.clip(0, width - 1)
        dply[:, :] = np.asarray(
            list(map(lambda p, q: p-self.hy[Y[p, q], X[p, q]]/self.zoom, Y, X)))
        dply.clip(0, height - 1)


        dplx = width-dplx  # mirror effect
        
        self.dplx = dplx
        self.dply = dply

    def apply(self, image):
        img = np.empty_like(image)
        img[:, :] = np.asarray(
            list(map(lambda p, q: image[self.dply[p, q], self.dplx[p, q]], self.Y, self.X)))
        return img


def clip(array, val_min, val_max):
    array[np.where(array < val_min)] = val_min
    array[np.where(array > val_max)] = val_max
    return array

if __name__ == "__main__":
    MORPH_FILE = "dpl_xy_z1_elliptical.npz"
    DEFAULT_CAM_ID = 1
    ZOOM = 0.07

    cam = Camera(DEFAULT_CAM_ID)
    morph = Morphing(cam.shape, MORPH_FILE, zoom=ZOOM)
    img_display = np.empty(cam.shape)

    while True:
        img_display[...] = morph.apply(
            cam.read_capture_device()
        )
        cam.show(img_display)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            cam.switch_capture_device()
            morph.set_target_shape(cam.shape)

        if cv2.waitKey(1) & 0xFF == ord('+'):
            morph.zoom += 0.01
            morph.init_morphing()

        if cv2.waitKey(1) & 0xFF == ord('-'):
            morph.zoom -= 0.01
            morph.init_morphing()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()
