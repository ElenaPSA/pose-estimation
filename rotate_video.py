import cv2

input_file = "Video_Bastien.mp4"
output_file = "rotated.mp4"

cap = cv2.VideoCapture(input_file)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Width and height are swapped after a 90-degree rotation
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, (height, width))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out.write(rotated)

cap.release()
out.release()

print(f"Video saved as: {output_file}")

