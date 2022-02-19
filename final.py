# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
import math
#DEBUG = False


"""
Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
V = [3, 5, 0]
Axis = [4, 4, 1]
Theta = 1.2 
print(np.dot(rotation_matrix(axis, theta), v))  
"""
def rotation_matrix(axis,theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# rotating the V around the Axis for Theta degree
def rotate(point, angle_degrees, axis=(0,1,0)):
    theta_degrees = angle_degrees
    theta_radians = math.radians(theta_degrees)
    rotated_point = np.dot(rotation_matrix(axis, theta_radians), point)
    return rotated_point

# clear the node info before adding new data
def clear_node():
    for n in tree.nodes:
        tree.nodes.remove(n)
        
    for n in tree.links:
        tree.links.remove(n)

# 
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

# lock camera to target with this fun
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty

SAMPLE = 2
RESOLUTION = 512
RESULTS_PATH = 'output'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0,0,0)
CIRCLE_FIXED_END = (.7,0,0)


fp = bpy.path.abspath(f"//{RESULTS_PATH}")

if not os.path.exists(fp):
    os.makedirs(fp)
    
    
scene = bpy.context.scene


# build light, SUN 
light_data = bpy.data.lights.new('Light', type='SUN')
light = bpy.data.objects.new('Light', light_data)
light.location = (1000, 1000, 1000)

# build camera 
cam_data = bpy.data.cameras.new('Camera')
cam = bpy.data.objects.new('Camera', cam_data)
cam.location = (20, 20, 20) #(x,y,z)


#link cam and light to the sence
bpy.context.collection.objects.link(cam)
bpy.context.collection.objects.link(light)
# assign the cam as render camera
scene.camera = cam

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}


# init a node tree for depth and normal information, Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
clear_node() # clear node tree
 
# Add passes for additionally dumping albedo and normals.
scene.view_layers["RenderLayer"].use_pass_normal = True
#scene.view_layers["ViewLayer"].use_pass_normal = True
scene.render.image_settings.file_format = str(FORMAT)
scene.render.image_settings.color_depth = str(COLOR_DEPTH)

#'''
if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapRange")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.inputs['From Min'].default_value = 0
        map.inputs['From Max'].default_value = 255
        map.inputs['To Min'].default_value = 1
        map.inputs['To Max'].default_value = 0
        #map = tree.nodes.new(type="CompositorNodeMapValue")
        #map.offset = [-0.7]
        #map.size = [DEPTH_SCALE]
        #map.use_min = True
        #map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
#'''

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True
    
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

# lock camera to the target
cam_constraint = cam.constraints.new(type='TRACK_TO') 
#cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
#cam_constraint.up_axis = 'UP_Y'
#cam_constraint.target = bpy.data.objects['10477_Satellite_v1_L3']
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

# Render Optimizations
scene.render.use_persistent_data = True
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.compute_device = 'NVIDIA GeForce RTX 2070'
#bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True


for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
    output_node.base_path = ''

out_data['frames'] = []

delta_angle = 360 / SAMPLE

for i in range(0, SAMPLE):
    # render base on current cam location
    cam_location = cam.location 
    #print(cam_location)
    #print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
    # file path
    scene.render.filepath = fp + '/r_' + str(i)
    tree.nodes['Depth Output'].file_slots[0].path = scene.render.filepath + "_depth_"
    tree.nodes['Normal Output'].file_slots[0].path = scene.render.filepath + "_normal_"
    #print(tree.nodes['Depth Output'].file_slots[0].path)
    
    bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': scene.render.filepath,
        'rotation': math.radians(delta_angle),
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)
    
    # update cam location
    new_cam_location = rotate(cam_location, delta_angle*i, axis=(0, 0, 1)) # along Z-axis
    cam.location = new_cam_location
    #b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
    #b_empty.rotation_euler[2] += radians(2*stepsize)

with open(fp + '/' + 'transforms.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)
