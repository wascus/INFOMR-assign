import numpy as np
import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def compute_normals(vertices, faces):
    normals = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        v0, v1, v2 = [vertices[idx] for idx in face]
        edge1 = np.array(v1) - np.array(v0)
        edge2 = np.array(v2) - np.array(v0)
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)  # Normalize
        normals[i] = normal
    return normals

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # vertex position
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):  # extracts the vertex indices and adds them to faces
                parts = line.split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # Only vertex indices
                faces.append(face)
    vertices = np.array(vertices)
    faces = np.array(faces)
    normals = compute_normals(vertices, faces)
    return vertices, faces, normals

def init_pygame(width=800, height=600):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for true 3D rendering
    gluPerspective(45, (width / height), 0.1, 50.0)  # Set up a perspective projection matrix
    glTranslatef(0.0, 0.0, -5)  # Move back to see the object
    glClearColor(1.0, 1.0, 1.0, 1.0)  # Set background to white

def init_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

def draw_obj(vertices, faces, normals):
    glBegin(GL_TRIANGLES)
    for i, face in enumerate(faces):
        glNormal3fv(normals[i])  # Set normal for each face
        for vertex_index in face:
            glVertex3fv(vertices[vertex_index])
    glEnd()

def draw_obj_with_wireframe(vertices, faces, normals, wireframe=False):
    if wireframe:
        # Draw edges only (wireframe mode)
        glDisable(GL_LIGHTING)  # No shading in wireframe mode
        glColor3f(0, 0, 0)  # Wireframe color (white)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        draw_obj(vertices, faces, normals)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)  # Re-enable lighting for shaded mode
    else:
        # Draw shaded model
        glColor3f(1.0, 1.0, 1.0)  # Wireframe color (white)
        glEnable(GL_LIGHTING)
        draw_obj(vertices, faces, normals)

def run_viewer(filename):
    vertices, faces, normals = load_obj(filename)
    init_pygame()
    init_lighting()

    rotation_x = 0
    rotation_y = 0
    wireframe = False
    zoom = -5
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    wireframe = not wireframe  # Toggle wireframe mode

        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            rotation_y -= 1
        if keys[K_RIGHT]:
            rotation_y += 1
        if keys[K_UP]:
            rotation_x -= 1
        if keys[K_DOWN]:
            rotation_x += 1

        # Clear screen and apply transformations
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glTranslatef(0.0, 0.0, zoom)  # Adjust zoom level
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        # Draw the object with or without wireframe
        draw_obj_with_wireframe(vertices, faces, normals, wireframe)

        glPopMatrix()
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    run_viewer('C:/Users/anest/Documents/GitHub/INFOMR-assign/test.obj')
