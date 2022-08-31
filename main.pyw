from virtualcam import VirtualCam
import cv2

options: dict[str, list[bool, str]] = {
    "freeze": [False, "f"],
    "tracking": [False, "g"],
    "focus": [False, "h"],
    "blur": [False, "b"],
    "mirror": [False, "m"],
}

try:
    vcam = VirtualCam((864, 480))
except ValueError as e:
    print(e)

while True:
    start = cv2.getTickCount()

    img = vcam.get_frame()

    key_pressed = vcam.get_key()

    if key_pressed is not None:
        for option, (flag, key_char) in options.items():
            if key_pressed == key_char:
                options[option][0] = not options[option][0]

        # Quit if ESC is pressed
        if key_pressed == "\x1b":
            break

    faces = vcam.detect_faces(img)

    if options["blur"][0]:
        img = cv2.blur(img, (15, 15))

    if options["focus"][0]:
        imgbuffer = vcam.blank_frame()
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Insert the face where the face was found
                imgbuffer[y : y + h, x : x + w] = img[y : y + h, x : x + w]

        img = imgbuffer

    if options["tracking"][0]:
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                vcam.draw_rectangle(img, (x, y, w, h))
                vcam.draw_text(img, "Face", (x, y - 5))

    if options["mirror"][0]:
        img = cv2.flip(img, 1)

    if not options["freeze"][0]:
        sentimg = img
        vcam.send(img)

    else:
        cv2.addWeighted(sentimg, 0.5, img, 0.5, 0, img)

        # Draw red text with white background on the text in the middle of the screen to indicate that the video is frozen
        middle = (vcam.width // 2, vcam.height // 2)
        vcam.draw_rectangle(
            img, (middle[0] - 100, middle[1] - 20, 230, 40), (0, 0, 0), True
        )
        vcam.draw_text(
            img, "Video frozen", (middle[0] - 100, middle[1] + 5), (0, 0, 255), 2
        )

    # Show the controls
    for i, (key, (flag, keychar)) in enumerate(options.items()):
        vcam.draw_text(
            img,
            f"{key.title()}: {keychar.upper()}",
            (10, vcam.height - (i + 1) * 20),
            (0, 255, 0) if flag else (0, 0, 255),
        )

    # Show the FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    vcam.draw_text(img, f"FPS: {int(fps)}", (10, 30))

    cv2.imshow("Video", img)
