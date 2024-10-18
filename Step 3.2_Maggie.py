import os
import open3d as o3d
import numpy as np
import csv
import pymeshlab

root_folder = 'ShapeDatabase_INFOMR-master-finalattempt' #folder with database (after hole filling)
output_csv = 'global_property_descriptors.csv' #file with results (saved as csv)


#function to compute volume using surface integrals
#cannot use built-in open3d or meshlab function because some meshes are not watertight
def compute_surface_integral_volume(mesh):
    volume = 0.0
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        v1 = vertices[tri[0]]
        v2 = vertices[tri[1]]
        v3 = vertices[tri[2]]

        normal = np.cross(v2 - v1, v3 - v1)  #calculate the normal of the triangle
        volume += np.dot(normal, v1)  #calculate the signed volume of the tetrahedron

    return abs(volume) / 6.0  #divide by 6 based on tetrahedron volume formula


#function to compute convex hull volume using pymeshlab (works with not watertight objects as well)
def compute_convex_hull_volume(mesh_path):
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_path)

        #apply the meshlab filter ‘Convex Hull’
        ms.apply_filter('generate_convex_hull') #calculate the convex hull with Qhull library (http://www.qhull.org/html/qconvex.htm)
        convex_hull_mesh = ms.current_mesh() #convex hull mesh

        volume = 0.0
        vertices = convex_hull_mesh.vertex_matrix()
        faces = convex_hull_mesh.face_matrix()

        #for each face (triangle), calculate the signed volume of the tetrahedron formed with the origin
        for face in faces:
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]

            normal = np.cross(v2 - v1, v3 - v1) #calculate the normal of the triangle
            volume += np.dot(normal, v1) #calculate the signed volume of the tetrahedron

        return abs(volume) / 6.0  #divide by 6 based on tetrahedron volume formula
    except Exception as e:
        print(f"Error occured when computing convex hull volume for object  {mesh_path}: {e}")
        return None


#compute global property descriptors
def compute_descriptors(mesh, mesh_path):
    descriptors = {}

    mesh.orient_triangles()  #reorient triangles so normals are consistently pointing outward
    mesh.compute_vertex_normals()

    #########################SURFACE AREA#############################
    descriptors['Surface Area'] = mesh.get_surface_area()
    ##################################################################

    #compute volume - used for compactness, rectangularity and convexity
    volume = compute_surface_integral_volume(mesh)
    descriptors['Volume'] = volume #save as a descriptor in case needed for testing


    ########################COMPACTNESS###############################
    if descriptors['Surface Area'] > 0:
        descriptors['Compactness'] = (36 * np.pi * (volume ** 2)) / (descriptors['Surface Area'] ** 3)
    else:
        descriptors['Compactness'] = None
    ##################################################################


    ##################### 3D RECTANGULARITY ##########################
    #compute bounding box volume - needed for rectangularity
    bounding_box = mesh.get_oriented_bounding_box()
    bounding_box_volume = bounding_box.volume()
    descriptors['Bounding Box Volume'] = bounding_box_volume

    if bounding_box_volume > 0:
        descriptors['Rectangularity'] = volume / bounding_box_volume
    else:
        descriptors['Rectangularity'] = None

    ##################################################################


    ########################DIAMETER##################################
    extents = bounding_box.extent
    descriptors['Diameter'] = np.max(extents)

    #CONVEXITY
    convex_hull_volume = compute_convex_hull_volume(mesh_path)
    if convex_hull_volume is not None and convex_hull_volume > 0:
        descriptors['Convexity'] = volume / convex_hull_volume
    else:
        descriptors['Convexity'] = None
    ##################################################################


    #######################ECCENTRICITY###############################
    points = np.asarray(mesh.vertices)
    cov_matrix = np.cov(points.T)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    if eigenvalues.min() > 0:
        descriptors['Eccentricity'] = eigenvalues.max() / eigenvalues.min()
    else:
        descriptors['Eccentricity'] = None

    return descriptors
    ##################################################################


#write everything to a csv file
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['Class', 'File', 'Surface Area', 'Volume', 'Compactness', 'Bounding Box Volume',
                  'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    #iterate over each class folder
    for class_name in os.listdir(root_folder):
        class_folder = os.path.join(root_folder, class_name)

        if os.path.isdir(class_folder): #make sure it is a directory
            for mesh_file in os.listdir(class_folder):
                if mesh_file.endswith('.obj'):
                    mesh_path = os.path.join(class_folder, mesh_file)

                    mesh = o3d.io.read_triangle_mesh(mesh_path) #use open3d to load a mesh

                    #make sure there is consistent orientation after hole filling
                    mesh.orient_triangles()  #reorient triangles consistently
                    mesh.compute_vertex_normals()  #recompute vertex normals

                    #compute all of the descriptors and save to the csv file
                    descriptors = compute_descriptors(mesh, mesh_path)
                    descriptors['Class'] = class_name
                    descriptors['File'] = mesh_file
                    writer.writerow(descriptors)

print(f"Global Property Descriptors saved to {output_csv}")

