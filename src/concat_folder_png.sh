
#concat a folder of pngs to a *.MTS
# ffmpeg -i tmp/img%09d.png \
# 	-y -c:v libx264 -filter:v fps=fps=60 \
# 	-pix_fmt hd720 mov/tmp/tmp.MTS


#concat a folder of pngs to a *.MTS
# ffmpeg -i frame_every_1_ms/img%09d.png \
# 	-y -c:v libx264 -filter:v fps=fps=60 \
# 	-f "scale=trunc(iw/2)*2:trunc(ih/2)*2,
# 	scale=1280:720:force_original_aspect_ratio=decrease,
# 	pad=1280:720:-1:-1:color=black
# 	" -pix_fmt hd720 tmp.MTS
# ffmpeg -i frame_every_1_ms/img%07d.png -y -c:v libx264 -filter:v fps=fps=30 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1:color=black" -pix_fmt yuv420p tmp.MTS

#this needs fixin' (downsamples framerate by alot).  bypassed for now...  it doesn't seem to add anything
# ffmpeg -i tmp.MTS -y -c:v libx264 -f -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1:color=black" -pix_fmt yuv420p out.mp4

# ffmpeg -i $INFN -y -c:v libx264 -r 30 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
#  -r 30 -pix_fmt yuv420p $TMP
# ffmpeg -i $TMP -y -q 0 -vf lutyuv=y='((val - minval)*255)/(maxval - minval)' -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1:color=black" $OUTFN

ffmpeg -i tmp/img%09d.png \
	-y -c:v libx264 -filter:v fps=fps=60 \
	-s hd720 ../vid/mov/tmp/tmp.MTS

#make text frames
source ../vid/mov/text.txt
# TEXT=$'Synchronous Variational Integrator\nDamped Neohookean Model\n(inward squish)'
# 1920x1080
ffmpeg -f lavfi -y -i color=black:1280x720 -f lavfi -i anullsrc  \
	-q 0 -vf drawtext="
	mov/arial.ttf:fontcolor=FFFFFF:fontsize=50:text=$TEXT:x=(main_w-text_w)/2:y=(main_h-text_h)/2,
	fade=t=in:st=0:d=0,
	fade=t=out:st=2:d=1,fps=fps=60
	" -c:v libx264 -b:v 1000k -s hd720 \
	-video_track_timescale 5000 -y -c:a aac -ar 0 \
	-ac 0 -sample_fmt fltp -t 4 ../vid/mov/tmp/intro.mp4
# ffmpeg -f lavfi -y -i color=black:1280x720 -f lavfi -i anullsrc -q 0 -vf drawtext="arial.ttf:fontcolor=FFFFFF:fontsize=50:text=$TEXT:x=(main_w-text_w)/2:y=(main_h-text_h)/2,fade=t=in:st=0:d=0,fade=t=out:st=2:d=1" -c:v libx264 -b:v 1000k -pix_fmt yuv420p -video_track_timescale 5000 -y -c:a aac -ar 0 -ac 0 -filter:v fps=fps=90 -sample_fmt fltp -t 4 intro.mp4
ffmpeg -i ../vid/mov/tmp/intro.mp4 -y -q 0 ../vid/mov/tmp/clip-1.MTS
ffmpeg -i ../vid/mov/tmp/tmp.MTS -y -q 0 ../vid/mov/tmp/clip-2.MTS
# ffmpeg -i intro.mp4 -y -q 0 clip-1.MTS
# ffmpeg -i tmp.MTS -y -q 0 clip-2.MTS

#concat the intro to the main clip
# ffmpeg -f concat -i mov/list.txt -y mov/tmp/tmp2.MTS
ffmpeg -f concat -i list.txt -c copy -y ../vid/mov/tmp/tmp2.MTS
# ffmpeg -f concat -i list.txt -c copy -y tmp2.MTS

#make it small enough to fit in an email
ffmpeg -i ../vid/mov/tmp/tmp2.MTS -y -c:v libx264 -crf 30 \
	-s hd720 out.mov
# ffmpeg -i tmp2.MTS -y -c:v libx264 -crf 30 -pix_fmt yuv420p out-1ms.mov
ffmpeg -i ../vid/out.mov -y ../vid/out.avi
#caution: slow
#motion blur without speeding up out-1ms.mov
# ffmpeg -i out-1ms.mov -filter:v "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=120'" fast-tips.mov
