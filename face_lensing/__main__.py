r"""
 _____                _                   _
|  ___|_ _  ___ ___  | |    ___ _ __  ___(_)_ __   __ _
| |_ / _` |/ __/ _ \ | |   / _ \ '_ \/ __| | '_ \ / _` |
|  _| (_| | (_|  __/ | |__|  __/ | | \__ \ | | | | (_| |
|_|  \__,_|\___\___| |_____\___|_| |_|___/_|_| |_|\__, |
                                                  |___/

An app to visualise your face through a gravitational lens.

This app works across platforms (Linux, macOS, Windows) and
had been created as a way to interrogate and facilitate questions
in science outreach events.

Developed by Alexandre Boucaud (Laboratoire APC, CNRS/IN2P3, Paris)
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

_BASEDIR = Path(__file__).parent.resolve()
LENS_FILE_PATH = _BASEDIR / "dpl_xy_z1_elliptical.npz"
DEFAULT_CAM = 0
DEFAULT_ZOOM = 0.07

TITLE = "Face Lensing"
COMMANDS = [
    "H : show/hide the app controls",
    "Q : shutdown app",
    "C : switch camera",
    "+ : increase lens effect",
    "- : decrease lens effect",
    "Space : take a screenshot",
]


class Camera:
    """Class handling i/o operations with the webcam and saving screenshots"""

    def __init__(self, cam_id=0, output_shape=None, output_dir=None):
        self.cam_id = cam_id
        self.output_shape = output_shape or (1280, 800)
        self.output_dir = output_dir or str(Path.cwd().resolve())
        self._init_app()

    def _init_app(self):
        self._save_counter = 0
        self._show_help = False
        self.set_capture_device()
        self.read_image_properties()

    def set_capture_device(self):
        self.camera = cv2.VideoCapture(self.cam_id)

    def read_capture_device(self):
        ack, img = self.camera.read()
        if not ack:
            raise ValueError("Image could not be read from device")
        return img

    def read_image_properties(self):
        img = self.read_capture_device()
        self.shape = img.shape[:2]

    def toggle_help(self):
        self._show_help = not self._show_help

    def show_commands(self, image):
        ypos = 50
        for i, cmd in enumerate(COMMANDS):
            cv2.putText(
                img=image,
                text=cmd,
                org=(30, ypos + 30 * i),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(240, 240, 240),
                thickness=2,
            )
            ypos += 30

    def show(self, image=None):
        image = image if image is not None else self.read_capture_device()
        image = cv2.resize(image, self.output_shape)
        if self._show_help:
            self.show_commands(image)
        cv2.imshow(TITLE, image)

    def switch_capture_device(self):
        self.cam_id = 1 - self.cam_id
        self.release()
        self.set_capture_device()
        self.read_image_properties()

    def take_screenshot(self, img):
        img_name = f"face_lensing_screenshot_{self._save_counter}.jpg"
        img_path = str(Path(self.output_dir) / img_name)
        cv2.putText(
            img=img,
            text="Made with Face Lensing - CNRS",
            org=(30, self.shape[0] - 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(240, 240, 240),
            thickness=2,
        )
        cv2.imwrite(img_path, img)
        self._save_counter += 1
        print(f"Image written as {img_path}")

    def release(self):
        self.camera.release()


class Morphing:
    """Class handling the morphing effects precomputed in a file"""

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
        lens_center = self.shape / 2
        img_center = self.target_shape / 2 + self.shift

        self.Y, self.X = np.indices(self.target_shape)

        # Create coordinates mapping between the image and the lens
        centred_coords = (
            np.indices(self.target_shape).reshape(2, -1)
            - img_center[:, None]
            + lens_center[:, None]
        ).astype(int)

        centred_coords[0] = centred_coords[0].clip(0, self.shape[0] - 1)
        centred_coords[1] = centred_coords[1].clip(0, self.shape[1] - 1)

        lens_coords = np.reshape(centred_coords, (2, *self.target_shape))

        dpl1y = map_coordinates(self.hy / self.zoom, lens_coords, order=1)
        dpl1x = map_coordinates(self.hx / self.zoom, lens_coords, order=1)

        self.dply = np.array(self.Y - dpl1y).astype(int)
        self.dplx = np.array(self.X - dpl1x).astype(int)
        # Mirror lens effect on the horizontal axis
        self.dplx = self.target_shape[1] - self.dplx

    def apply(self, image):
        return np.asarray(
            list(
                map(
                    lambda p, q: image[self.dply[p, q], self.dplx[p, q]], self.Y, self.X
                )
            )
        )

    def increase_effect(self):
        if self.zoom > 0.005:
            self.zoom -= 0.005
        self.init_morphing()

    def decrease_effect(self):
        self.zoom += 0.005
        self.init_morphing()


def main(lens_file=LENS_FILE_PATH, cam_id=DEFAULT_CAM, zoom=DEFAULT_ZOOM):
    print(__doc__)
    cam = Camera(cam_id)
    morph = Morphing(cam.shape, lens_file, zoom)
    img_display = cam.read_capture_device()

    while True:
        img_display[...] = morph.apply(cam.read_capture_device())
        cam.show(img_display)

        if keypress := cv2.waitKey(25):
            if keypress == ord("c"):
                print(f"Switching to camera {1 - cam.cam_id}")
                cam.switch_capture_device()
                morph.set_target_shape(cam.shape)
                img_display = cam.read_capture_device()

            if keypress == ord("+"):
                print("Increasing lens effect")
                morph.increase_effect()

            if keypress == ord("-"):
                print("Decreasing lens effect")
                morph.decrease_effect()

            if keypress == ord(" "):
                print("Taking a screenshot")
                cam.take_screenshot(img_display)

            if keypress == ord("h"):
                print("Showing/hiding commands")
                cam.toggle_help()

            if keypress == ord("q"):
                print("Quitting")
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
