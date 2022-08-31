import cv2
import pyvirtualcam as pvc
import numpy as np


class VirtualCam(pvc.Camera):
    """
    ### Virtual Cam

    This is the main class for the virtual cam. It is responsible for making pvc.Camera higher level and providing helper functions.
    """

    def __init__(
        self,
        size: tuple[int, int] = (864, 480),
        fps: int = 30,
        window_name: str = "Video",
    ) -> None:
        super().__init__(
            size[0], size[1], fps, backend="unitycapture", fmt=pvc.PixelFormat.BGR
        )  # Change backend to "obs" if you want to use OBS Virtual Camera instead.
        self.name = window_name
        self.size = size
        self._imcap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self._imcap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._imcap.set(cv2.CAP_PROP_FPS, fps)
        cv2.namedWindow(self.name)

    def blank_frame(self) -> np.ndarray:
        """
        Get a blank frame.
        """
        return np.zeros((self.height, self.width, 3), np.uint8)

    def get_frame(self) -> np.ndarray:
        """
        Get a frame from the camera.
        """
        success, frame = self._imcap.read()
        return frame if success else self.blank_frame()

    def draw_text(
        self,
        img: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 255),
        size: int = 1.5,
    ) -> None:
        """
        Draw text on the frame.
        """
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_PLAIN, size, color, 2)

    def draw_rectangle(
        self,
        img: np.ndarray,
        rect: tuple[int, int, int, int],
        color=(0, 0, 255),
        solid: bool = False,
    ):
        (x, y, w, h) = rect
        if solid:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    def detect_faces(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in the frame.
        """
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        ).detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)

    def get_key(self) -> str | None:
        """
        Get a key press.
        """
        raw_keycode = cv2.waitKey(1)
        return chr(raw_keycode % 256) if raw_keycode != -1 else None

    def close(self) -> None:
        """
        Close the camera.
        """
        self._imcap.release()
        cv2.destroyWindow(self.name)
        super().close()
