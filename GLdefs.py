
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import copy
import numpy as np
import pandas as pd
import gzip
import pickle
from pprint import pprint
from math import cos, sin
from time import time
from ArcBall import * 				# ArcBallT and this tutorials set of points/vectors/matrix types
import wafer
from math import pi as PI
PI2 = 2.0*3.1415926535			# 2 * PI (not squared!) 		// PI Squared
PI_3 = PI/3.

global layer
layer = 7

global allwafers
allwafers = None

global tc_map 
tc_map = None

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


g_Transform = Matrix4fT ()
g_LastRot = Matrix3fT ()
g_ThisRot = Matrix3fT ()

g_ArcBall = ArcBallT (640, 480)
g_isDragging = False
g_quadratic = None


keys = [False] * 1024   # Keypress states

def LoadGeometry(inF = "data/cell_map.pklz"):
    with gzip.open("data/cell_map.pklz", "rb") as f:
        _cell_map = pickle.load(f)
        _tc_map = pickle.load(f)


    branches = ["tc_x", "tc_y", "tc_z"]
    cell_map = _cell_map.query("tc_layer==%d and tc_zside == 1 and tc_subdet == 3" % layer)[branches]

    #tc_branches = ["x","y","z",'triggercell','neighbor_zside', 'neighbor_subdet','neighbor_layer','neighbor_wafer','neighbor_cell','neighbor_distance']
    #tc_map = _tc_map.query("layer == %d and zside == 1 and subdet == 3" % layer)[tc_branches]
    tc_branches = ["x","y","z",'triggercell']

    global tc_map
    tc_map = _tc_map.loc[1,3,layer][tc_branches]

    z = tc_map.iloc[0]["z"]

    r = (tc_map["x"]**2 + tc_map["y"]**2)**0.5
    tc_map["eta"] = np.arcsinh(tc_map["z"]/r)
    tc_map["phi"] = 2. * np.arctan(tc_map["y"] / (tc_map["x"] + r))

    tc_branches.append("eta")
    tc_branches.append("phi")

    return cell_map,tc_map


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
def Initialize (Width, Height):				# We call this right after our OpenGL window is created.
    global g_quadratic

    glClearColor(0.0, 0.0, 0.0, 1.0)					# This Will Clear The Background Color To Black
    glClearDepth(1.0)									# Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LEQUAL)								# The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)								# Enables Depth Testing
    glShadeModel (GL_FLAT);								# Select Flat Shading (Nice Definition Of Objects)
    glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST) 	# Really Nice Perspective Calculations

    g_quadratic = gluNewQuadric();
    gluQuadricNormals(g_quadratic, GLU_SMOOTH);
    gluQuadricDrawStyle(g_quadratic, GLU_FILL); 
    # Why? this tutorial never maps any textures?! ? 
    # gluQuadricTexture(g_quadratic, GL_TRUE);			# // Create Texture Coords

    glEnable (GL_LIGHT0)
    glEnable (GL_LIGHTING)

    glEnable (GL_COLOR_MATERIAL)


    # Load HGCal geometry
    cell_map, tc_map = LoadGeometry("data/cell_map.pklz")

    return True



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

def Hexagon(radius):
    glBegin(GL_TRIANGLE_FAN);
    for t in xrange(6):
        sintheta = sin(PI_3 * t)
        costheta = cos(PI_3 * t)

        glVertex3f (radius * costheta, radius * sintheta, 0.)
        glNormal3f (0., 0., 1.)
    glEnd()
    return

def HexagonXY(x,y,radius):
    glBegin(GL_TRIANGLE_FAN);
    for t in xrange(6):
        sintheta = sin(PI_3 * t)
        costheta = cos(PI_3 * t)

        glVertex3f (radius * (costheta+x), radius * (sintheta+y), 0.)
        glNormal3f (0., 0., 1.)
    glEnd()
    return

def DrawWafers():
    global allwafers
    global hexagons
    global cells
    global tc_map
    if allwafers is None:
        allwafers = tc_map.index.drop_duplicates().values
        wafer_x = {}
        wafer_y = {}
        wafer_z = {}
        global hexagons
        hexagons = [] 
        global cells
        cells = []
        #cells = np.ndarray(shape=(3,), dtype=np.float32)
        for w in allwafers:
            #x,y = tc_map["x"], tc_map["y"]
            wafer_x[w],wafer_y[w],wafer_z[w] = tc_map.loc[w]["x"], tc_map.loc[w]["y"], tc_map.loc[w]["z"]
            
            for i in xrange(len(wafer_x[w])):
                if len(cells) == 0:
                    cells = np.ndarray(shape=(3,), dtype=np.float32, buffer=np.array([wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]]))
                else:
                #cells = np.ndarray( [wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]], dtype=np.float32 )
                #else:
                    cells = np.append(cells, [wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]] )
                #cells.append( (wafer_x[w].iloc[i], wafer_y[w].iloc[i], wafer_z[w].iloc[i]) )
            hexagons.append( (wafer_x[w].mean(), wafer_y[w].mean(), wafer_z[w].mean()) )
        
        #pprint(hexagons)
        #pprint(wafer_x)
        #pprint(wafer_y)
   
    for i,h in enumerate(hexagons):
    #for h in np.nditer(cells):
#        glLoadIdentity();	    # // Reset The Current Modelview Matrix
        #glTranslatef(-1.5,0.0,-6.0);	 // Move Left 1.5 Units And Into The Screen 6.0
        
        #glTranslatef(h[0]/10.,h[1]/10., -10.0);
        #if i > 0: glTranslatef(-hexagons[i-1][0]/10.,-hexagons[i-1][1]/10., +10.0);
        #glTranslatef(h[0]/10.,h[1]/10., -10.0);
        if i > 0:
            glTranslatef((h[0]-hexagons[i-1][0])/10.,(h[1]-hexagons[i-1][1])/10., 0.);
        else: 
            glTranslatef(h[0]/10.,h[1]/10., -15.0);
        
        #glTranslatef(h[0],h[1], -600.0);

#        glPushMatrix();	# // NEW: Prepare Dynamic Transform
#        glMultMatrixf(g_Transform); # // NEW: Apply Dynamic Transform
        #glColor3f(0.75,0.75,1.0);
        #glColor3f(0.05,0.75,0.75);
        glColor4f(0.05,0.75,0.75,0.2);
        #Torus(0.30,1.00);
        Hexagon(0.6);
        #Hexagon(6.);
       # DrawWafers()
#        glPopMatrix();  # // NEW: Unapply Dynamic Transform

    return


def Draw ():
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
    glLoadIdentity();	    # // Reset The Current Modelview Matrix
    #glTranslatef(-1.5,0.0,-6.0);	 // Move Left 1.5 Units And Into The Screen 6.0
    glTranslatef(0.0,0.0,-25.0);

    glPushMatrix();	# // NEW: Prepare Dynamic Transform
    glMultMatrixf(g_Transform); # // NEW: Apply Dynamic Transform
    #glColor3f(0.75,0.75,1.0);
    glColor3f(0.05,0.75,0.75);
    #Torus(0.30,1.00);
    #Hexagon(1.);
   # DrawWafers()
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    DrawWafers()
    glPopMatrix();  # // NEW: Unapply Dynamic Transform

    glDisable(GL_BLEND);
    
    glLoadIdentity();	# // Reset The Current Modelview Matrix
    glTranslatef(1.5,0.0,-6.0);	 # // Move Right 1.5 Units And Into The Screen 7.0

    glPushMatrix();	    # // NEW: Prepare Dynamic Transform
    glMultMatrixf(g_Transform);	# // NEW: Apply Dynamic Transform
    glColor3f(1.0,0.75,0.75);
    #gluSphere(g_quadratic,1.3,20,20);
    glPopMatrix();	    # // NEW: Unapply Dynamic Transform

    # Draw text
    glut_print(100,700, text="Merry Christmas!", r=1.0)    
    glut_print(400,700, text="FPS: %.1f"%fps, r=0.1,g=1.0,b=0.1)    

    glFlush ();		    # // Flush The GL Rendering Pipeline
    glutSwapBuffers()
    return

