from moviepy.editor import VideoFileClip


def process_video(input_img):
    return input_img  # return input,result and mask heatmap image combined as one

output = 'tmp/tmp.mp4'
clip = VideoFileClip("data/dance_short.mp4")

# A get around for the issue https://github.com/Zulko/moviepy/issues/682
# solved by https://github.com/Zulko/moviepy/issues/586
if clip.rotation == 90:
    clip = clip.resize(clip.size[::-1])       # 90 degree rotate: (x, y) -> (y, x)
    clip.rotation = 0

clip = clip.fl_image(process_video)           # Call process_video to swap face for every frame in the video
clip.write_videofile(output, audio=False)

