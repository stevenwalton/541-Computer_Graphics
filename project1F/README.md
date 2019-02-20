# Creating a movie
```bash
ffmpeg -r 30 -i Images/frame%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p project1F.mp4
```
