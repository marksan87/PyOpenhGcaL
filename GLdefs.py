from __future__ import print_function
from __future__ import division
from camera.camera import Camera
from camera.spatial import Spatial
from camera.vecmath import * 
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import sys
import copy
import numpy as np
import pandas as pd
import gzip
import pickle
from numpy import ndarray
from pprint import pprint
from math import cos, sin
from time import time
from ArcBall import * 				# ArcBallT and this tutorials set of points/vectors/matrix types
import wafer
import math
from math import pi as PI
from globals import *
import config
#from config import subdet as subdet
#from config import _keys as _keys
PI2 = 2.0*PI			# 2 * PI (not squared!) 		
PI_3 = PI/3.
PI_75 = PI / 75.
PI_4 = PI / 4.
DEGREE = u"\xB0"
ESCAPE = '\033'
try:
    ESCAPEPY3 = b"\x1b"
except:
    ESCAPEPY3 = "000"

numlayers = {3:28,4:12} # Subdet:numlayers (1 to value)
hgcalDataFile = "hgcalDataForGL.pklz"
zside = 1

#global subdet
subdet = 4 

global allwafers
allwafers = None

global tc_map 
tc_map = None

global df_wafer
df_wafer = None

global waferColors
waferColors = {} 

global fps
fps = 0.0

global frame
frame = 0 

global nframe   # Frames to average over
nframe = 10     

global frameStartTime
frameStartTime = time()  

global frameEndTime
frameEndTime = 0.

global frameElapsedTime
frameElapsedTime = 0.

global aspectRatio
aspectRatio = 4./3.


global g_CameraM
g_CameraM = None

#global _keys

config._keys = dict()
config._keys["w"] = False 
config._keys["a"] = False
config._keys["s"] = False
config._keys["d"] = False
config._keys[GLUT_KEY_UP] = False
config._keys[GLUT_KEY_DOWN] = False
config._keys[GLUT_KEY_LEFT] = False
config._keys[GLUT_KEY_RIGHT] = False
config._keys["1"] = False
config._keys["2"] = False
config._keys["3"] = False
config._keys["4"] = False
config._keys["5"] = False
config._keys["-"] = False
config._keys["="] = False
config._keys["["] = False
config._keys["]"] = False
config._keys[ESCAPE] = False
#for i in xrange(256):
#    _keys[i] = False


g_Transform = Matrix4fT ()
g_LastRot = Matrix3fT ()
g_ThisRot = Matrix3fT ()

g_ArcBall = ArcBallT (640, 480)
g_isDragging = False
g_quadratic = None



#keys = [False] * 1024   # Keypress states

def LoadGeometry(inF = "data/cell_map.pklz"):
    with gzip.open("data/cell_map.pklz", "rb") as f:
        try:
            _cell_map = pickle.load(f)
            _tc_map = pickle.load(f)
        except UnicodeDecodeError:
            _cell_map = pickle.load(f, encoding='latin1')
            _tc_map = pickle.load(f, encoding='latin1')


    branches = ["tc_x", "tc_y", "tc_z"]
    cell_map = _cell_map.query("tc_layer==%d and tc_zside == 1 and tc_subdet == 3" % layer)[branches]

    #tc_branches = ["x","y","z",'triggercell','neighbor_zside', 'neighbor_subdet','neighbor_layer','neighbor_wafer','neighbor_cell','neighbor_distance']
    #tc_map = _tc_map.query("layer == %d and zside == 1 and subdet == 3" % layer)[tc_branches]
    tc_branches = ["x","y","z",'triggercell']

    global tc_map
    tc_map = _tc_map
    #tc_map = _tc_map.loc[1,3,layer][tc_branches]

#    z = tc_map.iloc[0]["z"]
#
#    r = (tc_map["x"]**2 + tc_map["y"]**2)**0.5
#    tc_map["eta"] = np.arcsinh(tc_map["z"]/r)
#    tc_map["phi"] = 2. * np.arctan(tc_map["y"] / (tc_map["x"] + r))
#
#    tc_branches.append("eta")
#    tc_branches.append("phi")

    return cell_map,tc_map

def LoadEventData(inF = "data/relvalQCD_df.pklz"):
    with gzip.open(inF, "rb") as f:
        df_perCell = pickle.load(f)
        df_perRoc = pickle.load(f)
        df_perWafer = pickle.load(f)
        df_gen = pickle.load(f)
        df_tower = pickle.load(f)

    return df_perWafer

texture = 0

def LoadTextures(imgF):
    #global texture
    try:
        # Hack for linux version of PIL
        Image.Image.tostring = Image.Image.tobytes
    except AttributeError:
        pass
    
    #image = imgOpen("NeHe.bmp")
    image = Image.open(imgF)
	
    ix = image.size[0]
    iy = image.size[1]
    try:
        image = image.tostring("raw", "RGBX", 0, -1)
    except AttributeError:
        image = image.tobytes("raw", "RGBX", 0, -1)


    # Create Texture	
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))   # 2d texture (x and y size)
	
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

def glut_print( x,  y,  font=GLUT_BITMAP_9_BY_15 ,  text="", r=0.,  g=0. , b=0. , a=1.):

    blending = False 
    if glIsEnabled(GL_BLEND) :
        blending = True

    #glEnable(GL_BLEND)
    glColor4f(r,g,b,a)
    glWindowPos2f(x,y)
    for ch in text :
        glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )


    if not blending :
        glDisable(GL_BLEND) 


# A general OpenGL initialization function.  Sets all of the initial parameters. 
def Initialize (Width, Height):
    global g_quadratic
    LoadTextures("CMSlogo.bmp")
    glClearColor(0.0, 0.0, 0.0, 1.0)					# This Will Clear The Background Color To Black
    glClearDepth(1.0)									# Enables Clearing Of The Depth Buffer

    glDepthFunc(GL_LEQUAL)								# The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)								# Enables Depth Testing
    glShadeModel (GL_FLAT);								# Select Flat Shading (Nice Definition Of Objects)
    glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST) 	# Really Nice Perspective Calculations

    # Make fullscreen
    if config.fullscreen:
        glutFullScreen();

    g_quadratic = gluNewQuadric();
    gluQuadricNormals(g_quadratic, GLU_SMOOTH);
    gluQuadricDrawStyle(g_quadratic, GLU_FILL); 
    # Why? this tutorial never maps any textures?! ? 
    # gluQuadricTexture(g_quadratic, GL_TRUE);			# // Create Texture Coords

    glEnable (GL_LIGHT0)
    glEnable (GL_LIGHTING)
    glEnable (GL_COLOR_MATERIAL)

    
    config.camera = Camera()
    sx,sy,sz = config.startPos
    config.player = Spatial(position = v3(sx,sy,sz))
    config.camera.set_position(config.player.get_position())
    config.camera.look_at(v3(0, 0, 0))
    config.camera.set_fov(PI/3)

    #glMatrixMode(GL_PROJECTION)
    #glLoadIdentity()
    #glPushMatrix(camera.get_projection_matrix())

    global hexagons
    global waferColors
    with gzip.open(hgcalDataFile, "rb") as f:
        try:
            hexagons = pickle.load(f)
            waferColors = pickle.load(f)
        except UnicodeDecodeError:
            hexagons = pickle.load(f, encoding='latin1')
            waferColors = pickle.load(f, encoding='latin1')

##################################################################
    # Load HGCal geometry
#    cell_map, tc_map = LoadGeometry("data/cell_map.pklz")
#
#    global hexagons
#    hexagons = {} 
#    for subdet,nlayers in numlayers.items():
#        wafer_x = {}
#        wafer_y = {}
#        wafer_z = {}
#        hexagons[subdet] = {}
#        for l in xrange(1,nlayers+1):
#            tc = tc_map.loc[zside,subdet,l][["x","y","z",'triggercell']]
#            allwafers = tc.index.drop_duplicates().values
#            hexagons[subdet][l] = []
#            for w in allwafers:
#                wafer_x[w],wafer_y[w],wafer_z[w] = tc.loc[w]["x"], tc.loc[w]["y"], tc.loc[w]["z"]
#                hexagons[subdet][l].append( [wafer_x[w].mean(), wafer_y[w].mean(), wafer_z[w].mean(), w] )
    ###############################################################
#    allwafers = tc_map.index.drop_duplicates().values
#    wafer_x = {}
#    wafer_y = {}
#    wafer_z = {}
#    global cells
#    cells = []
#
#    for w in allwafers:
#        #x,y = tc_map["x"], tc_map["y"]
#        wafer_x[w],wafer_y[w],wafer_z[w] = tc_map.loc[w]["x"], tc_map.loc[w]["y"], tc_map.loc[w]["z"]
#
#        for i in xrange(len(wafer_x[w])):
#            if len(cells) == 0:
#                cells = np.ndarray(shape=(3,), dtype=np.float32, buffer=np.array([wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]]))
#            else:
#            #cells = np.ndarray( [wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]], dtype=np.float32 )
#            #else:
#                cells = np.append(cells, [wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]] )
#            #cells.append( (wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]) )
#        hexagons.append( (wafer_x[w].mean(), wafer_y[w].mean(), wafer_z[w].mean(), w) )


    global df_wafer
#    df_wafer = LoadEventData("data/relvalQCD_df.pklz")
    

    #selectEvent(event)  
    return True

# Reshape The Window When It's Moved Or Resized
def ReSizeGLScene(Width, Height):
    #print "Resizing screen!"

    if Height == 0:                                             # Prevent A Divide By Zero If The Window Is Too Small 
        Height = 1

    glViewport(0, 0, Width, Height)             # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)                 # // Select The Projection Matrix
    glLoadIdentity()                                    # // Reset The Projection Matrix
    # // field of view, aspect ratio, near and far
    # This will squash and stretch our objects as the window is resized.
    # Note that the near clip plane is 1 (hither) and the far plane is 1000 (yon)
    aspectRatio = float(Width)/float(Height)
    #gluPerspective(45.0, apsectRatio, 1, 1000.0)
    config.camera.set_aspect(aspectRatio)
    glMultMatrixf(config.camera.get_projection_matrix()) 
    glMatrixMode (GL_MODELVIEW);                # // Select The Modelview Matrix
    glLoadIdentity ();                                  # // Reset The Modelview Matrix
    g_ArcBall.setBounds (Width, Height) # //*NEW* Update mouse bounds for arcball
    
    
    #global screenW
    #global screenH
    #screenW = Width
    #screenH = Height


    
#    print "New width:  %d" % screenW
#    print "New height: %d" % screenH
#    print "New ratio = %.2f" % aspectRatio

    return



#def selectEvent(evt=0):
#    global waferColors
#    global hexagons
#    global df_wafer

#    if evt not in waferColors:
#        waferColors[evt] = {}
#        for s,maxl in numlayers.items():
#            waferColors[evt][s] = {}
#            for l in xrange(1,maxl+1):
#                waferColors[evt][s][l] = {}
#
#                df = df_wafer.query("event == %d and tc_subdet == %d and tc_layer == %d" % (evt,s,l) )[["tc_wafer","tc_energy"]]
#                energyMax = df["tc_energy"].max()
#                hexLayer = l if s == 3 else l-numlayers[3]
#                for i,h in enumerate(hexagons[s][hexLayer]):
#                    # Fill colors
#                    wafer = h[3]
#                    try:
#                        energy = df.query("tc_wafer==%d" % wafer)["tc_energy"].values[0]
#                        waferColors[evt][s][l][wafer] = [energy/energyMax, 0.1, 0.1, 1.0]
#                    except IndexError:
#                        # Wafer not found, set default color 
#                        waferColors[evt][s][l][wafer] = [0.05,0.75,0.75,0.2]


#    return
    


def Mouse_PassiveDrag (mouse_dx, mouse_dy):
    """ Mouse cursor is moving but no buttons are pressed
    """
    #print "mouse_dx =", mouse_dx
    #print "mouse_dy =", mouse_dy
    look_speed = .2
    buffer = glGetDoublev(GL_MODELVIEW_MATRIX)
    c = (-1 * np.mat(buffer[:3,:3]) * \
        np.mat(buffer[3,:3]).T).reshape(3,1)
    # c is camera center in absolute coordinates, 
    # we need to move it back to (0,0,0) 
    # before rotating the camera
    glTranslate(c[0],c[1],c[2])
    m = buffer.flatten()
    glRotate(mouse_dx * look_speed, m[1],m[5],m[9])
    glRotate(mouse_dy * look_speed, m[0],m[4],m[8])
    
    # compensate roll
    glRotated(-math.atan2(-m[4],m[5]) * \
        57.295779513082320876798154814105 ,m[2],m[6],m[10])
    glTranslate(-c[0],-c[1],-c[2])
#######################################################
#    global g_LastRot, g_Transform, g_ThisRot
#
#    mouse_pt = Point2fT (mouse_dx, mouse_dy)
#    ThisQuat = g_ArcBall.drag (mouse_pt)                                            # // Update End Vector And Get Rotation As Quaternion
#    g_ThisRot = Matrix3fSetRotationFromQuat4f (ThisQuat)            # // Convert Quaternion Into Matrix3fT
#    # Use correct Linear Algebra matrix multiplication C = A * B
#    g_ThisRot = Matrix3fMulMatrix3f (g_LastRot, g_ThisRot)          # // Accumulate Last Rotation Into This One
#    g_Transform = Matrix4fSetRotationFromMatrix3f (g_Transform, g_ThisRot)  # // Set Our Final Transform's Rotation From This One
#
    config.player.yaw(-mouse_dx * 0.005)
    config.player.pitch(-mouse_dx * 0.005)


    return

def KeySpecialPressed(key, x, y):
    config._keys[key] = True

    if key == GLUT_KEY_F11:
        config.fullscreen = not config.fullscreen
        if config.fullscreen:
            # Switch to fullscreen
            glutFullScreen()
        else:
            # Switch to windowed mode
            glutReshapeWindow(config.screenW, config.screenH)
            ReSizeGLScene(config.screenW, config.screenH)
    
    elif key == GLUT_KEY_F5:
        config.alphaExp = max(config.alphaExp - 0.05, 0.0)
    elif key == GLUT_KEY_F6:
        #config.alphaExp = min(config.alphaExp + 0.05, 1.0)
        config.alphaExp += 0.05 


def KeySpecialUp(key, x, y):
    config._keys[key] = False 

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def KeyPressed(*args):
    global g_quadratic
        
    key = str(args [0])
    #print ("str(args[0]) =", str(args[0]))
    if key[0] == "b":
        # Remove python3 string formatting characters
        key = str(key[2:-1])

   
    
    if key == "-":# and ("-" not in _keys or _keys["-"] == False):
        # Key - just pressed
        config.event -= 1
        if config.event < 0:
            config.event = 9

        #selectEvent(event) 
			
    elif key == "=":# and ("=" not in _keys or _keys["="] == False):
        # Key = just pressed
        config.event += 1
        if config.event > 9:
            config.event = 0

        #selectEvent(event) 

    

    elif key == "3":# and ("3" not in config._keys or config._keys["3"] == False):
        config.subdet = 3
    
    elif key == "4":# and ("4" not in _keys or _keys["4"] == False):
        config.subdet = 4

    elif key == "5":# and ("5" not in _keys or _keys["5"] == False):
        config.subdet = 0

    elif key == "r":# and ("r" not in _keys or _keys["r"] == False):
        # Reset player and camera positions
        sx,sy,sz = config.startPos
        config.player.set_position(v3(sx,sy,sz))
        config.camera.set_position(config.camera.get_position())
        config.camera.set_fov(PI_3)
        config.layerSpacing = 0.5

    elif key == "[":
        config.layerSpacing = max(0.1, config.layerSpacing - 0.1)
    
    elif key == "]":
        config.layerSpacing += 0.1
   
    elif key == "h":
        config.hitsOnly = not config.hitsOnly

    elif key == "`":
        config.debugDisplay = not config.debugDisplay 


    config._keys[key] = True

    #print("key == %s: " % ESCAPEPY3, key == ESCAPEPY3)
    #if "ESCAPE" in _keys or ESCAPE in _keys:
    if key == ESCAPE or args[0] == ESCAPEPY3:# or config._keys[ESCAPE]:
        gluDeleteQuadric (g_quadratic)
        sys.exit ()
    
    #UpdateKeys()
    return

# The function called whenever a key is released. Note the use of Python tuples to pass in: (key, x, y)  
def KeyUp(*args):
    global g_quadratic
    # If escape is pressed, kill everything.
    key = str(args [0])
    #print ("str(args[0]) =", str(args[0]))
    if key[0] == "b":
        # Remove python3 string formatting characters
        key = str(key[2:-1])
    config._keys[key] = False 
    
    #UpdateKeys()
    return
    


def UpdateKeys():
    #global keys
    global g_quadratic
    #if ESCAPE in config._keys:
    #    gluDeleteQuadric (g_quadratic)
    #    sys.exit ()
    
    #global _keys
    fwd = 0.
    strafe = 0.
    speed = 0.1

#    global event
#    if "2" in _keys:
#        if event == 9:
#            event = 0
#        else:
#            event += 1
#
#        selectEvent(event)
#
#    if "1" in _keys:
#        if event == 0:
#            event = 9
#        else:
#            event -= 1
#
#        selectEvent(event)
    
    if config._keys["w"] or config._keys[GLUT_KEY_UP]:
        fwd += speed
    if config._keys["s"] or config._keys[GLUT_KEY_DOWN]:
        fwd -= speed
    if config._keys["a"] or config._keys[GLUT_KEY_LEFT]:
        strafe += speed
    if config._keys["d"] or config._keys[GLUT_KEY_RIGHT]:
        strafe -= speed

    x,y,z = config.player.get_position()
    cx,cy,cz = config.camera.get_position()

#    fwd = .1 * (  if "w" in _keys)-_keys["s"])
#    strafe = .1 * (_keys["a"]-_keys["d"])
    if abs(fwd) or abs(strafe):
        m = glGetDoublev(GL_MODELVIEW_MATRIX).flatten()
        #print "glTranslate(%f, %f, %f)" % (fwd*m[2],fwd*m[6],fwd*m[10])
        config.player.set_position(v3(x+fwd*m[2]+strafe*m[0],y+fwd*m[6]+strafe*m[4],z+fwd*m[10]+strafe*m[8]))    
        config.camera.set_position(v3(cx+fwd*m[2],cy+fwd*m[6],cz+fwd*m[10]))   
        config.camera.look_at(v3(0.,0.,0.))
#        glTranslate(fwd*m[2],fwd*m[6],fwd*m[10])
#        glTranslate(strafe*m[0],strafe*m[4],strafe*m[8])
 
    return

def UpdateCamera():
    up = v3(*config.player.get_world_up())
    position = v3(*config.player.get_position())
    #position += (v3(*player.get_world_forward()) * -10.0)
    #position += (up * 3.0)

    config.camera.set_position(position)
    config.camera.set_orientation(config.player.get_orientation())

    global g_Transform, g_CameraM

    g_CameraM = Matrix3fSetIdentity();
#    _cameraM = Matrix3fSetRotationFromQuat4f(camera.get_orientation())
#    
#    g_CameraM = Matrix4fSetRotationFromMatrix3f(g_CameraM,_cameraM)
    g_CameraM = config.camera.get_camera_matrix()

def Upon_Drag (cursor_x, cursor_y):
    """ Mouse cursor is moving
            Glut calls this function (when mouse button is down)
            and pases the mouse cursor postion in window coords as the mouse moves.
    """
    global g_isDragging, g_LastRot, g_Transform, g_ThisRot

    if (g_isDragging):
        mouse_pt = Point2fT (cursor_x, cursor_y)
        ThisQuat = g_ArcBall.drag (mouse_pt)						# // Update End Vector And Get Rotation As Quaternion
        g_ThisRot = Matrix3fSetRotationFromQuat4f (ThisQuat)		# // Convert Quaternion Into Matrix3fT
        # Use correct Linear Algebra matrix multiplication C = A * B
        g_ThisRot = Matrix3fMulMatrix3f (g_LastRot, g_ThisRot)		# // Accumulate Last Rotation Into This One
        g_Transform = Matrix4fSetRotationFromMatrix3f (g_Transform, g_ThisRot)	# // Set Our Final Transform's Rotation From This One
    return

def Upon_Click (button, button_state, cursor_x, cursor_y):
    """ Mouse button clicked.
            Glut calls this function when a mouse button is
            clicked or released.
    """
    global g_isDragging, g_LastRot, g_Transform, g_ThisRot

    g_isDragging = False
    if (button == GLUT_RIGHT_BUTTON and button_state == GLUT_UP):
        # Right button click
        g_LastRot = Matrix3fSetIdentity ();							# // Reset Rotation
        g_ThisRot = Matrix3fSetIdentity ();							# // Reset Rotation
        g_Transform = Matrix4fSetRotationFromMatrix3f (g_Transform, g_ThisRot);	# // Reset Rotation
    elif (button == GLUT_LEFT_BUTTON and button_state == GLUT_UP):
        # Left button released
        g_LastRot = copy.copy (g_ThisRot);							# // Set Last Static Rotation To Last Dynamic One
    elif (button == GLUT_LEFT_BUTTON and button_state == GLUT_DOWN):
        # Left button clicked down
        g_LastRot = copy.copy (g_ThisRot);							# // Set Last Static Rotation To Last Dynamic One
        g_isDragging = True											# // Prepare For Dragging
        mouse_pt = Point2fT (cursor_x, cursor_y)
        g_ArcBall.click (mouse_pt);								# // Update Start Vector And Prepare For Dragging

    return

def mouseWheel(button, direction, x, y):
    if direction > 0.:
        # Zooming in
        config.camera.set_fov(max(config.camera.get_fov()-PI_75,PI_75))
    else:
        config.camera.set_fov(min(config.camera.get_fov()+PI_75,PI))

def Torus(MinorRadius, MajorRadius):		
    # // Draw A Torus With Normals
    glBegin( GL_TRIANGLE_STRIP );									# // Start A Triangle Strip
    for i in xrange (20): 											# // Stacks
        for j in xrange (-1, 20): 										# // Slices
            # NOTE, python's definition of modulus for negative numbers returns
            # results different than C's
            #       (a / d)*d  +  a % d = a
            if (j < 0):
                wrapFrac = (-j%20)/20.0
                wrapFrac *= -1.0
            else:
                wrapFrac = (j%20)/20.0;
            phi = PI2*wrapFrac;
            sinphi = sin(phi);
            cosphi = cos(phi);

            r = MajorRadius + MinorRadius*cosphi;

            glNormal3f (sin(PI2*(i%20+wrapFrac)/20.0)*cosphi, sinphi, cos(PI2*(i%20+wrapFrac)/20.0)*cosphi);
            glVertex3f (sin(PI2*(i%20+wrapFrac)/20.0)*r, MinorRadius*sinphi, cos(PI2*(i%20+wrapFrac)/20.0)*r);

            glNormal3f (sin(PI2*(i+1%20+wrapFrac)/20.0)*cosphi, sinphi, cos(PI2*(i+1%20+wrapFrac)/20.0)*cosphi);
            glVertex3f (sin(PI2*(i+1%20+wrapFrac)/20.0)*r, MinorRadius*sinphi, cos(PI2*(i+1%20+wrapFrac)/20.0)*r);
    glEnd();														# // Done Torus
    return

def DrawSquare(xoffset=0.1,yoffset=0.1, xsize = 0.4):
    #(-1,-1): bottom left corner of screen
    #(1,1): top right corner

    glEnable(GL_TEXTURE_2D)
    glColor3f( 1, 1, 1 )
    glBegin( GL_QUADS )
    
    ysize = xsize * config.camera.get_aspect() 
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.+xoffset,  1.-yoffset-ysize,  -1.0);	# Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.+xoffset+xsize,  1.-yoffset-ysize,  -1.0);	# Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.+xoffset+xsize,  1.-yoffset,  -1.0);	# Top Right Of The Texture and Quad
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.+xoffset,  1.-yoffset,  -1.0);	# Top Left Of The Texture and Quad
	
#    glTexCoord2f(0.0, 0.0); glVertex3f(-0.9,  0.0,  0.0);	# Bottom Left Of The Texture and Quad
#    glTexCoord2f(1.0, 0.0); glVertex3f( 0.0,  0.0,  0.0);	# Bottom Right Of The Texture and Quad
#    glTexCoord2f(1.0, 1.0); glVertex3f( 0.0,  0.867,  0.0);	# Top Right Of The Texture and Quad
#    glTexCoord2f(0.0, 1.0); glVertex3f(-0.9,  0.867,  0.0);	# Top Left Of The Texture and Quad
	
    glEnd(  )

    glDisable(GL_TEXTURE_2D)


def Hexagon(radius):
    glBegin(GL_TRIANGLE_FAN);
    for t in range(6):
        sintheta = sin(PI_3 * t)
        costheta = cos(PI_3 * t)

        glVertex3f (radius * costheta, radius * sintheta, 0.)
        glNormal3f (0., 0., 1.)
    glEnd()
    return

def HexagonXYZ(xoffset=0.,yoffset=0.,zoffset=0.,radius=0.6):
    glBegin(GL_TRIANGLE_FAN);
    for t in range(6):
        sintheta = sin(PI_3 * t)
        costheta = cos(PI_3 * t)

        glVertex3f (radius * (costheta+xoffset), radius * (sintheta+yoffset), zoffset)
        glNormal3f (0., 0., 1.)
    glEnd()
    return

def DrawWafers(_subdet=3, _layer=7, xoffset = 0., yoffset = 0., zoffset = 0., color = None):
    global allwafers
    global hexagons
    global cells
    global tc_map
    global df_wafer
    global waferColors
    event = config.event

    glPopMatrix()
    glLoadIdentity();	    # // Reset The Current Modelview Matrix
    UpdateKeys()
    UpdateCamera()
    #glTranslatef(10.0,0.0,-20.0);
    x,y,z = config.camera.get_position()
    glTranslatef(x,y,z);
    glPushMatrix();	# // NEW: Prepare Dynamic Transform
    glMultMatrixf(g_Transform); # // NEW: Apply Dynamic Transform
    # Multiply view (camera) matrix

    #df = df_wafer.query("event == %d" % event)[["tc_wafer","tc_energy"]]
    #energyMax = df["tc_energy"].max()
    
    #print df_wafer["tc_energy"].max()["tc_energy"].max()
#    if allwafers is None:
#        allwafers = tc_map.index.drop_duplicates().values
#        wafer_x = {}
#        wafer_y = {}
#        wafer_z = {}
#        global hexagons
#        hexagons = [] 
#        global cells
#        cells = []
#        #cells = np.ndarray(shape=(3,), dtype=np.float32)
#        for w in allwafers:
#            #x,y = tc_map["x"], tc_map["y"]
#            wafer_x[w],wafer_y[w],wafer_z[w] = tc_map.loc[w]["x"], tc_map.loc[w]["y"], tc_map.loc[w]["z"]
#            
#            for i in xrange(len(wafer_x[w])):
#                if len(cells) == 0:
#                    cells = np.ndarray(shape=(3,), dtype=np.float32, buffer=np.array([wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]]))
#                else:
#                #cells = np.ndarray( [wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]], dtype=np.float32 )
#                #else:
#                    cells = np.append(cells, [wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]] )
#                #cells.append( (wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]) )
#            hexagons.append( (wafer_x[w].mean(), wafer_y[w].mean(), wafer_z[w].mean(), w) )

        #pprint(hexagons)
        #pprint(wafer_x)
        #pprint(wafer_y)

    waferLayer = _layer
#    if _subdet == 4: 
#        waferLayer = _layer - 28
#    elif _subdet == 3:
#        waferLayer = _layer

    if _subdet == 4:
        _layer -= 28


    scaling = 15.
    for i,h in enumerate(hexagons[_subdet][_layer]):
#        if i > 0:
#            glTranslatef((h[0]-hexagons[_subdet][_layer][i-1][0])/10.,(h[1]-hexagons[_subdet][_layer][i-1][1])/10., -(h[2]-hexagons[_subdet][_layer][i-1][2])/5.);
#        else: 
#            glTranslatef(h[0]/10.,h[1]/10., -h[2]/5.);
        
        if i > 0:
            glTranslatef((h[0]-hexagons[_subdet][_layer][i-1][0])/scaling,(h[1]-hexagons[_subdet][_layer][i-1][1])/scaling, 0.);
        else: 
            glTranslatef(h[0]/scaling+xoffset,h[1]/scaling+yoffset, zoffset);

        #x,y,z = h[0]/scaling+xoffset, h[1]/scaling+yoffset, zoffset
        # Determine color based on energy
        wafer = h[3]

        if color is not None:
            r,g,b,a = color
        else:
            r,g,b,a = waferColors[event][_subdet][waferLayer][wafer]
            hexZoffset = r/5.
        if a < 1.0:
            if config.hitsOnly:
                # Don't draw cells with no energy deposited
                continue
            a = 0.05    # adjust alpha of background wafers
        elif r > 0.:
            #a = max(1./(r**0.5),1.0)    # adjust alpha of signal wafers
            a = max(r**config.alphaExp,0.0)    # adjust alpha of signal wafers
        glColor4f(r,g,b,a)
#        try:
#            energy = df.query("tc_wafer==%d" % wafer)["tc_energy"].values[0]
#            glColor4f(energy/energyMax, 0.1, 0.1, 1.0)
#        except IndexError:
#            # Wafer not found, set default color 
#            glColor4f(0.05,0.75,0.75,0.2);
        #Torus(0.30,1.00);
        #glColor4f(0.05,0.75,0.75,0.2);
        HexagonXYZ(zoffset=hexZoffset,radius=0.6);
        #HexagonXYZ(x,y,z,0.6);
        #Hexagon(6.);
       # DrawWafers()i
#        glPopMatrix();  # // NEW: Unapply Dynamic Transform


    return


def Draw ():
    # Pseudo-global variables
    subdet = config.subdet

    global fps
    global frame
    global frameStartTime
    global frameEndTime
    global frameElapsedTime
    frame += 1  # Increment frame counter
    
    frameEndTime = time()
    frameElapsedTime = frameEndTime - frameStartTime

    if frameElapsedTime > 1.0:
        fps = frame / frameElapsedTime
        frame = 0
        frameStartTime = time()

#    if frame % nframe == 0:
#        frameEndTime = time()
#        frame = 0
#        fps = nframe / (frameEndTime - frameStartTime)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	 # // Clear Screen And Depth Buffer
   
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Draw cms logo
    DrawSquare(xoffset = 0.,yoffset=0.,xsize = 0.2)
    
    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix();

    glLoadIdentity()
    glMultMatrixf(config.camera.get_projection_matrix())
    #glLoadMatrix(camera.get_projection_matrix())
    #glUniformMatrixf(camera.get_camera_matrix())
    
    
    glMatrixMode(GL_MODELVIEW) 
    glLoadIdentity();	    # // Reset The Current Modelview Matrix
    #glTranslatef(-1.5,0.0,-6.0);	 // Move Left 1.5 Units And Into The Screen 6.0
    
    
    
    
#    glTranslatef(10.0,0.0,-25.0);

    glPushMatrix();	# // NEW: Prepare Dynamic Transform
    glMultMatrixf(g_Transform); # // NEW: Apply Dynamic Transform
    UpdateCamera()
    #glColor3f(0.75,0.75,1.0);
    glColor3f(0.05,0.75,0.75);
    #Torus(0.30,1.00);
    #Hexagon(1.);
   # DrawWafers()
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    global layer
	
#    for l in xrange(1,numlayers[subdet]-1,2):
#        DrawWafers(_subdet=subdet, _layer=l, xoffset = -0., yoffset = 0., zoffset = -15. + 0.5*(l - 5))
    #global subdet
    if subdet == 0:
        # Display subdet 3 and 4
        for i,l in enumerate(range(1,41,2)):
            #DrawWafers(_subdet=(3 if l <= numlayers[3] else 4), _layer=l, xoffset = -0., yoffset = 0., zoffset = 0. + config.layerSpacing*(l - 5))
            DrawWafers(_subdet=(3 if l <= numlayers[3] else 4), _layer=l, xoffset = -0., yoffset = 0., zoffset = -config.layerSpacing * 10. + float(i)*config.layerSpacing)
    
    elif subdet == 3:
        for i,l in enumerate(range(1,29,2)):
            #DrawWafers(_subdet=subdet, _layer=l, xoffset = -0., yoffset = 0., zoffset = -20. + config.layerSpacing*(l - 5))
            DrawWafers(_subdet=subdet, _layer=l, xoffset = -0., yoffset = 0., zoffset = -config.layerSpacing * 7. + float(i)*config.layerSpacing)
    
    elif subdet == 4:
        for i,l in enumerate(range(29,41,2)):
            DrawWafers(_subdet=subdet, _layer=l, xoffset = -0., yoffset = 0., zoffset = -config.layerSpacing * 3. + float(i)*config.layerSpacing)

    global g_CameraM
    glMultMatrixf(g_CameraM)
    #for l in xrange(1,29,2):
    #    DrawWafers(_subdet=subdet, _layer=l, xoffset = -0., yoffset = 0., zoffset = -15. + 0.5*(l - 14))

###    if layer > 3:
###        DrawWafers(_subdet=subdet, _layer=(layer-4),zpos=-1.)
###    
###        if layer > 1:
###            #DrawWafers(_subdet=subdet, _layer=(layer-2),zpos=-3.,color = (0.,1.,0.,1.))
###            DrawWafers(_subdet=subdet, _layer=(layer-2),zpos=-3.)
####        glPopMatrix()
####        glLoadIdentity();	    # // Reset The Current Modelview Matrix
####        glTranslatef(0.0,0.0,-25.0);
####        glPushMatrix();	# // NEW: Prepare Dynamic Transform
####        glMultMatrixf(g_Transform); # // NEW: Apply Dynamic Transform
###    #DrawWafers(_subdet=subdet, _layer=layer,zpos=-5.,color=(244./255.,244./255.,64./255.,1.))
###    DrawWafers(_subdet=subdet, _layer=layer,zpos=-5.)
###    if layer < 27:
###        #DrawWafers(_subdet=subdet, _layer=(layer+2),zpos=-7.,color = (0.,0.,1.,1.))
###        DrawWafers(_subdet=subdet, _layer=(layer+2),zpos=-7.)
###        if layer < 25:
###            DrawWafers(_subdet=subdet, _layer=(layer+4),zpos=-9.)
    glPopMatrix();  # // NEW: Unapply Dynamic Transform


    
    glDisable(GL_BLEND);
    
    

#    glLoadIdentity();	# // Reset The Current Modelview Matrix
#    glTranslatef(1.5,0.0,-6.0);	 # // Move Right 1.5 Units And Into The Screen 7.0
#
#    glPushMatrix();	    # // NEW: Prepare Dynamic Transform
#    glMultMatrixf(g_Transform);	# // NEW: Apply Dynamic Transform
#    glColor3f(1.0,0.75,0.75);
#    #gluSphere(g_quadratic,1.3,20,20);
#    glPopMatrix();	    # // NEW: Unapply Dynamic Transform

    # Draw text
    #global event

    m_viewport = np.array([0]*4,dtype=np.int32)
    glGetIntegerv( GL_VIEWPORT, m_viewport );
    screenW = m_viewport[2]
    screenH = m_viewport[3]
    x,y,z = config.camera.get_position()
    #glut_print(100,700, text="Merry Christmas! Event %d  Subdet %d layer %d" % (event,subdet,layer), r=1.0)    
    glut_print(100,550, text="Happy New Year! RelVal QCD Event %d" % config.event, r=1.0)    
#    glut_print(60,400, text=" Width: %d" % screenW, r=1.0)    
#    glut_print(60,380, text="Height: %d" % screenH, r=1.0)    
#    global aspectRatio
#    glut_print(60,360, text="Ratio = %.2f" % aspectRatio, r = 1.0)
#    
    #global subdet
    if subdet == 3:
        glut_print(100,525, text="Subdet 3: EE", r=0.1,g=1.0,b=0.1)    
    elif subdet == 4:
        glut_print(100,525, text="Subdet 4: FH", r=0.1,g=1.0,b=0.1)    
    elif subdet == 0:
        glut_print(100,525, text="Subdet 3+4 (EE+FH)", r=0.1,g=1.0,b=0.1)    
    else: 
        glut_print(100,525, text="Subdet %d" % subdet , r=0.1,g=1.0,b=0.1)    
    
    if config.debugDisplay:
        glut_print(100,500, text="FPS: %.1f"%fps, r=0.1,g=1.0,b=0.1)    
        glut_print(100,475, text="Camera pos: (%.2f, %.2f, %.2f)" % (x,y,z), g=1.0)    
        glut_print(100,450, text="FOV: %.1f%s" % (config.camera.get_fov()*180./PI, DEGREE), r=0.1,g=1.0,b=0.1)
        glut_print(100,425, text="Spacing: %.1f" % config.layerSpacing, r=0.1,g=1.0,b=0.1)
        glut_print(100,400, text="hitsOnly: %s" % config.hitsOnly, r=0.1,g=1.0,b=0.1)
        glut_print(100,375, text="alphaExp: %.2f" % config.alphaExp, r=0.1,g=1.0,b=0.1)
    

    glFlush ();		    # // Flush The GL Rendering Pipeline
    glutSwapBuffers()
    
    
    return

