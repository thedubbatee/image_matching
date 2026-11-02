import cv2
from harris import HarrisLaplaceDetector
from dog import DoGDetector

image = cv2.imread("image.jpg")

for name, detector in (("harris", HarrisLaplaceDetector()), ("dog", DoGDetector())):
    keypoints = detector.detect(image)

    print(f"Detected {len(keypoints)} {name} keypoints")

    # Create a copy so the original image is unchanged
    output = image.copy()

    # Draw a circle at each detected keypoint
    for kp in keypoints:
        x = int(round(kp.x))
        y = int(round(kp.y))

        cv2.circle(
            output,
            center=(x, y),
            radius=max(2, int(round(3 * kp.sigma))),
            color=(0, 0, 255),  # Red (OpenCV uses BGR)
            thickness=2
        )

    # Save the annotated image
    cv2.imwrite(f"detected_corners_{name}.png", output)

# Display the result
cv2.waitKey(0)
cv2.destroyAllWindows()