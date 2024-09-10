import numpy as np
import random
import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def load_obj(filename):
    """
    reads a 3D model in the OBJ file format,
    extracts vertex positions and face data, and assigns a random colour to each face.
    
    output: A tuple of three elements:
      -  vertices
      -  faces (each face consists of indices referring to the vertices)
      -  colors (randomly generated colours, one for each face)
    """
    vertices = []
    faces = []
    colors = []  
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # vertex position
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):  # extracts the vertex indices and adds them to faces
                parts = line.split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # Only vertex indices
                faces.append(face)
                # Assign a random color for this face
                colors.append([random.random(), random.random(), random.random()])
    return np.array(vertices), faces, colors



def init_pygame(width=800, height=600):
    """
    sets up Pygame and configures OpenGL for rendering
    """
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL) 
    glEnable(GL_DEPTH_TEST) # Enable depth testing for true 3D rendering
    gluPerspective(45, (width / height), 0.1, 50.0) # Set up a perspective projection matrix
    glTranslatef(0.0, 0.0, -5)  # Move back to see the object


def draw_obj(vertices, faces, colors):
    """
    draws the 3D model using OpenGL
    """
    glBegin(GL_TRIANGLES)
    for i, face in enumerate(faces):
        glColor3f(*colors[i])  # Use the pre-generated color for the face
        for vertex_index in face:
            glVertex3fv(vertices[vertex_index])
    glEnd()

def run_viewer(filename):
    vertices, faces, colors = load_obj(filename)
    init_pygame()

    rotation_x = 0
    rotation_y = 0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            rotation_y -= 1
        if keys[K_RIGHT]:
            rotation_y += 1
        if keys[K_UP]:
            rotation_x -= 1
        if keys[K_DOWN]:
            rotation_x += 1

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        draw_obj(vertices, faces, colors)

        glPopMatrix()
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    run_viewer(sys.argv[1])
