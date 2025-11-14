To generate the video of the network learning the circle, use this ffmpeg command from the root level dir:

```
ffmpeg -framerate 60 -i cmd/circle/imgs/frame_%03d.png -pix_fmt yuv420p circle_learning.mp4
```
