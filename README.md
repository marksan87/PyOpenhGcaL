Visualization of HGCal wafer energies using OpenGL and compatible with python 2 or 3

## Prerequisites
- Python 2.7 or 3.3+ 
- PyOpenGL: Detailed instructions can be found [here](http://pyopengl.sourceforge.net/documentation/installation.html)
```
pip install PyOpenGL PyOpenGL_accelerate Pillow numpy pandas

```
 

- Display with OpenGL 2+ support. OpenGL is installed by default for Windows. Debian Linux systems will need to install the packages for GLE and GLUT: 
```
sudo apt install libgle3 freeglut3-dev
```

## Instructions
To run the simulation:

```
python wafer.py
```

### Controls

Movement:
w,s,a,d  or  up,down,left,right
Click and drag to rotate 


Event toggle (down,up):
\- or =  

Subdetector display:
3   (EE)
4   (FH)
5   (EE and FH)


