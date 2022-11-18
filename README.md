Face Lensing
============

[![License][license-badge]](LICENSE) ![Python supported versions][pyversion-badge] [![PyPI][pypi-badge]][pypi]

___Look at yourself through a gravitational lens___

<img alt="Face Lensing screenshot example" src="example2.jpg" height=450px>

This program was developed for **outreach purposes**.

It deforms the input feed of a webcam as if the light had gone through a gravitational lens.  
While the shape of the lens was precomputed and cannot be changed, one can play with the intensity of the deformation (using <kbd>+</kbd> / <kbd>-</kbd> keys) to make interesting effects appear.

The effect is best seen from a certain distance, when the head is well centered with respect to the webcam, but appears on the screen in multiple places around the image center.

Feel free to install and play with it, it has been tested on Windows, Linux and macOS.

Installation
------------

1. Install the Python program with `pip`
    ```sh
    pip install face-lensing
    ```
2. Launch the app from a terminal
    ```sh
    face-lensing-app
    ```
3. A program should appear with the distorted webcam image


App controls
------------

The program can be controlled with a keyboard using the following keys

|       Key        | Action                                      |
| :--------------: | ------------------------------------------- |
|   <kbd>h</kbd>   | show/hide command helper                    |
|   <kbd>q</kbd>   | quit the program                            |
|   <kbd>c</kbd>   | change camera (in case of multiple webcams) |
| <kbd>Space</kbd> | save a screenshot locally                   |
|   <kbd>+</kbd>   | increase the lensing effect                 |
|   <kbd>-</kbd>   | decrease the lensing effect                 |


Acknowledgements
----------------

The original idea and precomputation of the lens deformation is attributed to Johan Richard (CRAL, CNRS/INSU)

License
-------

This program is licensed under the [MIT license](LICENSE).

[license-badge]: https://img.shields.io/github/license/aboucaud/face_lensing?color=blue
[pyversion-badge]: https://img.shields.io/pypi/pyversions/face_lensing?color=yellow&logo=pypi
[pypi-badge]: https://badge.fury.io/py/face_lensing.svg
[pypi]: https://pypi.org/project/face_lensing/
