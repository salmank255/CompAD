import ffmpeg
import os

def convert_to_mp4(mkv_file,videos_dir):
    print(mkv_file)
    no_extension = str(os.path.splitext(mkv_file)[0])
    with_mp4 = no_extension + ".mp4"
    print(with_mp4)
    ffmpeg.input(videos_dir+'/'+mkv_file).output(videos_dir+'/'+with_mp4).run()
    print("Finished converting {}".format(no_extension))


videos_dir = 'mkv_webm_vids'
videofiles = os.listdir(videos_dir)
videofiles = [af for af in videofiles if not af.endswith('.mp4')]
for i, videofile in enumerate(videofiles):
        print('\n %d videofile '%i, videofile, '\n')
        convert_to_mp4(videofile,videos_dir)