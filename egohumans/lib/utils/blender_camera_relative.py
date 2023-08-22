# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import re
import bpy
import sys
import argparse
import numpy as np
from math import radians
from mathutils import Matrix, Quaternion


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())

## in rgb
def get_colors(c):
    colors = {
        'pink': np.array([197, 27, 125]),
        'light_pink': np.array([233, 163, 201]),
        'light_green': np.array([161, 215, 106]),
        'green': np.array([77, 146, 33]),
        'red': np.array([215, 48, 39]),
        'light_red': np.array([252, 146, 114]),
        'light_orange': np.array([252, 141, 89]),
        'purple': np.array([118, 42, 131]),
        'light_purple': np.array([175, 141, 195]),
        'light_blue': np.array([145, 191, 219]),
        'blue': np.array([69, 117, 180]),
        'gray': np.array([130, 130, 130]),
        'white': np.array([255, 255, 255]),
        'turkuaz': np.array([50, 134, 204]),
        'yellow': np.array([235, 219, 52]),
        # 'orange': np.array([242, 151, 43]),
        'orange': np.array([220, 88, 42]),

    }
    return colors[c]

##################################################
# Helper functions
##################################################

# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
#   Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


##################################################
# Set camera extrinsics from OpenCV camera parameters
#
# NOTE: THIS CODE DOES NOT WORK PROPERLY WHEN BOTH rt and t !=0
#       It is better to apply translation+rotation to objects and keep camera at default looking along Y
#
##################################################
def set_camera_extrinsics_from_opencv(camera, rt, t):
    # Map OpenCV translation to Blender coordinates
    x = t[0]
    y = t[2]
    z = -t[1]

    # OpenCV applies translation to all vertices
    # In Blender we need to move camera in opposite direction to achieve same effect
    camera.location = (-x, -y, -z)

    mat = Rodrigues(np.array(rt))

    # Apply rotation first to Blender camera system (Looking down negative world Z-Axis)
    quat = Matrix(mat).to_quaternion()
    quat = Quaternion((-quat.w, quat.x, -quat.y, -quat.z))

    camera.rotation_mode = 'QUATERNION'

    # Map rotation in Blender camera system to world rotation
    # Default camera looking along positive Y axis with XY being the ground plane

    quat_world_from_camera = Quaternion((1.0, 0.0, 0.0), radians(90))
    camera.rotation_quaternion = quat_world_from_camera @ quat


##################################################
# Set camera intrinsics from OpenCV camera parameters
##################################################
# https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
def set_camera_intrinsics_from_opencv(camera, width, height, fx, fy, cx, cy):
    # Resolution
    bpy.data.scenes['Scene'].render.resolution_x = width
    bpy.data.scenes['Scene'].render.resolution_y = height

    # Use horizontal sensor fit
    camera.data.sensor_fit = 'HORIZONTAL'

    # Focal length
    camera.data.lens_unit = 'MILLIMETERS'
    sensor_width_in_mm = camera.data.sensor_width
    camera.data.lens = (fx / width) * sensor_width_in_mm

    # TODO: fx != fy

    # Camera principal point
    #
    # Note: For nonrectangular images a shift_y of 0.5 is not moving the resulting image by half the height
    #       shift_y=1 is the same pixel shift as shift_x=1
    #       (Blender 2.79b)
    camera.data.shift_x = -(cx / width - 0.5)
    camera.data.shift_y = (cy - 0.5 * height) / width

    pixel_aspect = fy / fx
    bpy.data.scenes['Scene'].render.pixel_aspect_x = 1.0
    bpy.data.scenes['Scene'].render.pixel_aspect_y = pixel_aspect

    return


def np_array_from_image(img_name):
    img = bpy.data.images.load(img_name, check_existing=True) # bpy.data.images[img_name]
    img = np.array(img.pixels[:])
    return img


def save_image(fname, img):
    output_image = bpy.data.images.new('save_img', height=img.shape[0], width=img.shape[1])
    output_image.file_format = 'PNG'
    output_image.pixels = img.ravel()
    output_image.filepath_raw = fname
    output_image.save()


def overlay_smooth(img, render):
    img = np_array_from_image(img)
    render = np_array_from_image(render)
    img_size = int(np.sqrt(render.shape[0] // 4))

    render = render.reshape((img_size, img_size, 4))
    img = img.reshape((img_size, img_size, 4))

    # breakpoint()

    m = render[:, :, -1:] #  / 255.
    i = img[:, :, :3] * (1 - m) + render[:, :, :3] * m
    i = np.clip(i, 0., 1.) # .astype(np.uint8)
    i = np.concatenate([i, np.zeros((img_size, img_size, 1))], axis=-1)
    return i


def look_at(camera, point):
    loc_camera = camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    camera.rotation_euler = rot_quat.to_euler()
    return

def set_camera_translation(camera, x, y, z):
    # OpenCV applies translation to all vertices
    # In Blender we need to move camera in opposite direction to achieve same effect
    camera.location = (x, y, z)

    return

def get_vertices_global(obj):
    verts = obj.data.vertices
    vlen = len(verts)
    obj_matrix = np.matrix(obj.matrix_world.inverted().to_3x3())
    
    verts_co_1D = np.zeros([vlen*3], dtype='f')
    verts.foreach_get("co", verts_co_1D)
    verts_co_3D = verts_co_1D.reshape(vlen, 3)
    
    verts_co_3D = verts_co_3D @ obj_matrix
    verts_co_3D += np.array(obj.location)

    return verts_co_3D


def render_mesh(object_paths, output_dir, output_path, colors, wireframe, quads, \
 width, height, rotation_angle=0, fx=5000, fy=5000, cx=5000, cy=5000):
    ####################
    # Object
    ####################

    for i, (object_path, color) in enumerate(zip(object_paths, colors)):
        print("Loading obj: " + object_path)
        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]

        mc = get_colors(color) / 255.

        bpy.data.materials['Body'].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*mc, 1)
        object.data.materials[0] = bpy.data.materials['Body'].copy()

        object.rotation_euler[2] = radians(rotation_angle)

        if quads:
            bpy.context.view_layer.objects.active = object
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.tris_convert_to_quads()
            bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.shade_smooth()

    # Mark freestyle edges
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')


    print('-----------------------------------------------------------------')
    ####################
    # Camera
    ####################
    # numpy_path = object_path.replace('.obj', '.npy')
    # print("Loading npy: " + numpy_path)

    camera = bpy.data.objects['Camera']

    camera_translation = (0, 0, 0)
    set_camera_extrinsics_from_opencv(camera, (0.0, 0.0, 0.0), camera_translation)

    scale = 1

    set_camera_intrinsics_from_opencv(camera, width, height, fx * scale, fy * scale, cx, cy)

    ####################
    # Render
    ####################
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)


    # Delete last selected object from scene
    object.select_set(True)
    bpy.ops.object.delete()

    return

##############################################################################
# Main
##############################################################################

if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('-i', '--inp', type=str, required=True, help='input directory')
    parser.add_argument('-o', '--out', type=str, default=None, help='output directory')
    parser.add_argument('-of', '--out_file', type=str, default=None, help='output file')
    parser.add_argument('-w', '--wireframe', action='store_true', help='draws quad wireframe')
    parser.add_argument('-t', '--thickness', type=float, default=0.15, help='wireframe thickness')
    parser.add_argument('-fx', '--fx', type=float, default=5000, help='camera focal length x')
    parser.add_argument('-fy', '--fy', type=float, default=5000, help='camera focal length y')
    parser.add_argument('-cx', '--cx', type=float, default=5000, help='camera center x')
    parser.add_argument('-cy', '--cy', type=float, default=5000, help='camera center y')
    parser.add_argument('-c', '--colors', type=str, default='turkuaz', help='mesh color')
    parser.add_argument('-sx', '--size_x', type=int, default=720, help='image size width')
    parser.add_argument('-sy', '--size_y', type=int, default=720, help='image size height')
    parser.add_argument('--sideview', action='store_true', help='flag to render side view meshes')
    parser.add_argument('--turntable', action='store_true', help='render with turntable')
    args = parser.parse_args()

    # argv = sys.argv
    # argv = argv[argv.index("--") + 1:]  # get all args after "--"

    print('Input arguments:', args)  # --> ['example', 'args', '123']

    if args.inp.endswith('.obj'):
        print('Processing a single file')
        input_file = args.inp
        input_file = os.path.abspath(input_file)
        filelist = [os.path.basename(args.inp)]
        input_dir = input_file.replace(os.path.basename(args.inp), '')
        output_dir = input_dir
    else:
        input_dir = args.inp
        output_dir = args.out if args.out else input_dir.replace('mesh_output', 'blender_output')
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # Process data in directory
        if args.sideview:
            filelist = [x for x in sorted(os.listdir(input_dir)) if x.endswith('.obj')]
        else:
            filelist = [x for x in sorted(os.listdir(input_dir)) if x.endswith('.obj') and not x.endswith('_rot.obj')]

    wireframe = args.wireframe
    debug = False

    # Render setup
    scene = bpy.data.scenes['Scene']
    scene.render.use_freestyle = wireframe
    scene.render.line_thickness = args.thickness

    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy

    # Change mesh color
    colors = args.colors.split('###')

    img_width = args.size_x
    img_height = args.size_y

    print('Num of files to be processed', len(filelist))

    mesh_fns = []
    for idx, input_file in enumerate(filelist):
        print(input_file)
        mesh_fn = os.path.join(input_dir, input_file)
        mesh_fns.append(mesh_fn)

    output_path = args.out_file
    render_mesh(
        mesh_fns,
        output_dir,
        output_path,
        colors=colors,
        wireframe=False,
        quads=True,
        width=img_width,
        height=img_height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

