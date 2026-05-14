

@dataclass
class DifferenceRegion:
    """Stores one difference rectangle and its found/revealed state."""

    x: int
    y: int
    width: int
    height: int
    alteration_name: str
    found: bool = False
    revealed: bool = False

    @property
    def centre(self) -> Tuple[int, int]:
        """Return the centre of the region in original image coordinates."""
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def radius(self) -> int:
        """Circle radius large enough to cover the region."""
        return int(math.hypot(self.width, self.height) / 2) + 8

    def overlaps(self, other: "DifferenceRegion", margin: int = 16) -> bool:
        """Return True when this region overlaps another region, using a margin."""
        return not (
            self.x + self.width + margin < other.x
            or other.x + other.width + margin < self.x
            or self.y + self.height + margin < other.y
            or other.y + other.height + margin < self.y
        )

    def contains_click(self, x: int, y: int, tolerance: int) -> bool:
        """Check whether a click is inside or near the hidden difference region."""
        return (
            self.x - tolerance <= x <= self.x + self.width + tolerance
            and self.y - tolerance <= y <= self.y + self.height + tolerance
        )

    def __str__(self) -> str:
        return (
            f"{self.alteration_name}: x={self.x}, y={self.y}, "
            f"w={self.width}, h={self.height}, found={self.found}"
        )

class ImageLoader:
    """Loads images robustly and normalises them into BGR format for OpenCV."""

    @staticmethod
    def load_bgr(path: str) -> np.ndarray:
        """
        Read an image as a BGR NumPy array.

        np.fromfile + cv2.imdecode is used so that Windows paths containing
        spaces or non-English characters are handled more safely than cv2.imread.
        """
        raw = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("The selected file could not be read as an image.")

        # Convert grayscale to BGR.
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convert transparent PNG/BGRA to BGR over a white background.
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3].astype(float) / 255.0
            white = np.full_like(bgr, 255)
            blended = (bgr.astype(float) * alpha[:, :, None]) + (
                white.astype(float) * (1.0 - alpha[:, :, None])
            )
            return blended.astype(np.uint8)

        if image.shape[2] == 3:
            return image

        raise ValueError("Unsupported image channel format.")
