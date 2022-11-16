Face Lensing
============

[![License][license-badge]](LICENSE) ![Python supported versions][pyversion-badge] [![PyPI][pypi-badge]][pypi]

Look at yourself through a gravitational lens.

Installation
------------

1. Install the Python program with `pip`
    ```sh
    pip install face_lensing
    ```
2. Launch the app from a terminal
    ```sh
    face-lensing-app
    ```
3. A window should appear with the distorted webcam image


App controls
------------

The app can be controlled through the keyboard using the following controls:

|       Key        | Action                                      |
| :--------------: | ------------------------------------------- |
|   <kbd>h</kbd>   | show/hide command helper                    |
|   <kbd>q</kbd>   | quit the program                            |
|   <kbd>c</kbd>   | change camera (in case of multiple webcams) |
| <kbd>Space</kbd> | save a screenshot locally                   |
|   <kbd>+</kbd>   | increase the lensing effect                 |
|   <kbd>-</kbd>   | decrease the lensing effect                 |


License
-------

This program is licensed under the [MIT license](LICENSE).

[license-badge]: https://img.shields.io/github/license/aboucaud/face_lensing?color=blue
[pyversion-badge]: https://img.shields.io/pypi/pyversions/face_lensing?color=yellow&logo=pypi
[pypi-badge]: https://badge.fury.io/py/face_lensing.svg
[pypi]: https://pypi.org/project/face_lensing/
