#!/usr/bin/env python
from __future__ import print_function
from argparse import ArgumentParser
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import pickle
import gzip
from GLdefs import *		# Draw (), Initialize () and all the real OpenGL work.
from ArcBall import *		# // *NEW* ArcBall header
import numpy as np
import pandas as pd
from GLdefs import *
from camera.camera import Camera
from camera.spatial import Spatial
from globals import *
import config

# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'

# Number of the glut window.
window = 0

# Camera object
global camera
camera = None

# Spatial object  
global player
player = None

global fov
fov = 45.0

global layer
layer = 8

#def LoadGeometry():
#    with gzip.open("data/cell_map.pklz", "rb") as f:
#        _cell_map = pickle.load(f)
#        _tc_map = pickle.load(f)
#
#
#    branches = ["tc_x", "tc_y", "tc_z"]
#    cell_map = _cell_map.query("tc_layer==%d and tc_zside == 1 and tc_subdet == 3" % layer)[branches]
#
#    #tc_branches = ["x","y","z",'triggercell','neighbor_zside', 'neighbor_subdet','neighbor_layer','neighbor_wafer','neighbor_cell','neighbor_distance']
#    #tc_map = _tc_map.query("layer == %d and zside == 1 and subdet == 3" % layer)[tc_branches]
#    tc_branches = ["x","y","z",'triggercell']
#    
#    global tc_map
#    tc_map = _tc_map.loc[1,3,layer][tc_branches]
#
#    z = tc_map.iloc[0]["z"]
#
#    r = (tc_map["x"]**2 + tc_map["y"]**2)**0.5
#    tc_map["eta"] = np.arcsinh(tc_map["z"]/r)
#    tc_map["phi"] = 2. * np.arctan(tc_map["y"] / (tc_map["x"] + r))
#
#    tc_branches.append("eta")
#    tc_branches.append("phi")
#
#    return cell_map,tc_map



# A general OpenGL initialization function.  Sets all of the initial parameters. 
##def InitGL(Width, Height):				# We call this right after our OpenGL window is created.
##
##    glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
##    glClearColor(0.0, 0.0, 0.0, 0.5)	# This Will Clear The Background Color To Black
##    glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
##    glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
##    glDepthFunc(GL_LEQUAL)				# The Type Of Depth Test To Do
##    glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST) # Really Nice Perspective Calculations
##
##    # Make fullscreen
##    #glutFullScreen();
##    
##    print "Foobar"
##    # Initialize camera
###    global camera
###    camera = Camera()
###    camera.set_position(v3(0, 0, -10))
###    camera.look_at(v3(0, 0, 0))
##
##     
##
##    return True									# // Initialization Went OK


    


# Reshape The Window When It's Moved Or Resized
###def ReSizeGLScene(Width, Height):
###    print "Resizing screen!"
###    
###    if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
###        Height = 1
###    
###    glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
###    glMatrixMode(GL_PROJECTION)			# // Select The Projection Matrix
###    glLoadIdentity()					# // Reset The Projection Matrix
###    # // field of view, aspect ratio, near and far
###    # This will squash and stretch our objects as the window is resized.
###    # Note that the near clip plane is 1 (hither) and the far plane is 1000 (yon)
###    gluPerspective(45.0, float(Width)/float(Height), 1, 100.0)
###
###    glMatrixMode (GL_MODELVIEW);		# // Select The Modelview Matrix
###    glLoadIdentity ();					# // Reset The Modelview Matrix
###    g_ArcBall.setBounds (Width, Height)	# //*NEW* Update mouse bounds for arcball
###    return


# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def keyPressed(*args):
    global g_quadratic
    # If escape is pressed, kill everything.
    key = args [0]
    if key == ESCAPE:
        gluDeleteQuadric (g_quadratic)
        sys.exit ()

    if key == KEY_UP or key == KEY_W:
        # move up
        pass        


def main(fullscreen = False):
    global window
    # pass arguments to init
    glutInit(sys.argv)

    if fullscreen:
        # Force full screen
        config.fullscreen = True
    #cell_map, tc_map = LoadGeometry()
    
    # Select type of Display mode:   
    #  Double buffer 
    #  RGBA color
    # Alpha components supported 
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    
    # get a 640 x 480 window 
    #glutInitWindowSize(640, 480)
    #glutInitWindowSize(1024, 768)
    glutInitWindowSize(config.screenW,config.screenH)
    
    # the window starts at the upper left corner of the screen 
    glutInitWindowPosition(0, 0)
    
    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python, remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main.
    global layer
    window = glutCreateWindow("CMS HGCal Wafer Energy")

    # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.	
    glutDisplayFunc(Draw)
    
    # Uncomment this line to get full screen.
    #glutFullScreen()

    # When we are doing nothing, redraw the scene.
    glutIdleFunc(Draw)
    
    # Register the function called when our window is resized.
    glutReshapeFunc(ReSizeGLScene)
    
    # Register the function called when the keyboard is pressed.  
    #glutKeyboardFunc(keyPressed)
    glutKeyboardFunc(KeyPressed)
    
    # Register the function called when a key is released 
    glutKeyboardUpFunc(KeyUp)

    # Register callback for when speical key (arrow, function, etc) is pressed
    glutSpecialFunc(KeySpecialPressed)    
    
    # Register callback for when speical key (arrow, function, etc) is released 
    glutSpecialUpFunc(KeySpecialUp)    

    # GLUT When mouse buttons are clicked in window
    glutMouseFunc (Upon_Click)
    
    # GLUT mouse wheel function
    glutMouseWheelFunc(mouseWheel);

    # GLUT When the mouse mvoes
    glutMotionFunc (Upon_Drag)
    #glutMotionFunc (Mouse_Motion)
    glutPassiveMotionFunc(Mouse_PassiveDrag)

    # We've told Glut the type of window we want, and we've told glut about
    # various functions that we want invoked (idle, resizing, keyboard events).
    # Glut has done the hard work of building up thw windows DC context and 
    # tying in a rendering context, so we are ready to start making immediate mode
    # GL calls.
    # Call to perform inital GL setup (the clear colors, enabling modes
    #Initialize (640, 480)
    #Initialize (1600, 1200)
    #Initialize (1024, 768, fullscreen)
    Initialize (config.screenW, config.screenH)

    # Start Event Processing Engine	
    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
if __name__ == "__main__":
    print("Hit ESC key to quit.")
    parser = ArgumentParser()
    parser.add_argument("-fs", dest="fullscreen", action="store_true", default=False, help="fullscreen mode")
    args = parser.parse_args()
    main(fullscreen = args.fullscreen)
    
