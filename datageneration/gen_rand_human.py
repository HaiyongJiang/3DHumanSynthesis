#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# File              : datageneration/gen_rand_human.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 03.08.2018
# Last Modified Date: 08.08.2018
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import sys
import os
import random
import math
import bpy
import numpy as np
from os.path import join, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam
from easydict import EasyDict as edict
import shutil
import cv2
import deepdish as dd
sys.path.insert(0, ".")


def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def dump(obj):
    print("###########################################")
    for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def crop_image(orig_img, cropped_min_x, cropped_max_x, cropped_min_y, cropped_max_y):
    '''Crops an image object of type <class 'bpy.types.Image'>.  For example, for a 10x10 image,
    if you put cropped_min_x = 2 and cropped_max_x = 6,
    you would get back a cropped image with width 4, and
    pixels ranging from the 2 to 5 in the x-coordinate

    Note: here y increasing as you down the image.  So,
    if cropped_min_x and cropped_min_y are both zero,
    you'll get the top-left of the image (as in GIMP).

    Returns: An image of type  <class 'bpy.types.Image'>
    '''
    num_channels=orig_img.channels
    #calculate cropped image size
    cropped_size_x = cropped_max_x - cropped_min_x
    cropped_size_y = cropped_max_y - cropped_min_y
    #original image size
    orig_size_x = orig_img.size[0]
    orig_size_y = orig_img.size[1]

    cropped_img = bpy.data.images.new(name="cropped_img", width=cropped_size_x, height=cropped_size_y)

    #loop through each row of the cropped image grabbing the appropriate pixels from original
    #the reason for the strange limits is because of the
    #order that Blender puts pixels into a 1-D array.
    current_cropped_row = 0
    for yy in range(orig_size_y - cropped_max_y, orig_size_y - cropped_min_y):
        #the index we start at for copying this row of pixels from the original image
        orig_start_index = (cropped_min_x + yy*orig_size_x) * num_channels
        #and to know where to stop we add the amount of pixels we must copy
        orig_end_index = orig_start_index + (cropped_size_x * num_channels)
        #the index we start at for the cropped image
        cropped_start_index = (current_cropped_row * cropped_size_x) * num_channels
        cropped_end_index = cropped_start_index + (cropped_size_x * num_channels)

        #copy over pixels
        cropped_img.pixels[cropped_start_index:cropped_end_index] = \
                                orig_img.pixels[orig_start_index:orig_end_index]

        #move to the next row before restarting loop
        current_cropped_row += 1
    return cropped_img

# image is not a numpy image
def shift_image(bg_img, v_trans):
    # note that the camera direction is -Y;
    x_start = int(v_trans[0])
    y_start = int(max(v_trans[1], v_trans[2]))
    img = bg_img[x_start:, y_start:]
    return img

def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None

sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(ob, params):
    materials = {}
    vgroups = {}
    with open('pkl/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = bpy.data.materials['Material'].copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        # assign active material; where we activate a material???
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return(materials)


def mv_result_files(params, idx_run=0, idx_frame=0, suffix="_0"):
    res_paths = {k:join(params['tmp_path'], '%05d_%s'%(idx_run, k))
                 for k in params['output_types'] if params['output_types'][k]}
    # render
    # vblur
    # depth
    # normal
    # segm
    for k in res_paths:
        path_k = res_paths[k] + "/" + "Image%04d.exr"%idx_frame
        if os.path.exists(path_k):
            os.rename(path_k, path_k.replace(".exr", "%s.exr"%suffix))
        elif os.path.exists(path_k.replace(".exr", ".png")):
            os.rename(path_k, path_k.replace(".exr", "%s.png"%suffix))


# create the different passes that we render
def create_composite_nodes(tree, params, res_paths, img=None, idx=0):
    ## .location ==> node position in the editor panel
    ## .xx ==> node attributes
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create node for foreground image
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400

    # create node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    bg_im.location = -300, 30
    if img is not None:
        bg_im.image = img

    # focus/ defocus???
    if(params['output_types']['vblur']):
    # create node for computing vector blur (approximate motion blur)
        vblur = tree.nodes.new('CompositorNodeVecBlur')
        vblur.factor = params['vblur']
        vblur.location = 240, 400

        # create node for saving output of vector blurred image
        vblur_out = tree.nodes.new('CompositorNodeOutputFile')
        vblur_out.format.file_format = 'PNG'
        vblur_out.base_path = res_paths['vblur']
        vblur_out.location = 460, 460

    # create node for mixing foreground and background images
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.location = 40, 30
    mix.use_alpha = True

    # create node for the final output
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30

    # create node for saving depth
    if(params['output_types']['depth']):
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        depth_out.location = 40, 700
        depth_out.format.file_format = 'OPEN_EXR'
        depth_out.base_path = res_paths['depth']

    # create node for saving normals
    if(params['output_types']['normal']):
        normal_out = tree.nodes.new('CompositorNodeOutputFile')
        normal_out.location = 40, 600
        normal_out.format.file_format = 'OPEN_EXR'
        normal_out.base_path = res_paths['normal']

    # create node for saving foreground image
    if(params['output_types']['fg']):
        fg_out = tree.nodes.new('CompositorNodeOutputFile')
        fg_out.location = 170, 600
        fg_out.format.file_format = 'PNG'
        fg_out.base_path = res_paths['fg']

    # create node for saving ground truth flow
    if(params['output_types']['gtflow']):
        gtflow_out = tree.nodes.new('CompositorNodeOutputFile')
        gtflow_out.location = 40, 500
        gtflow_out.format.file_format = 'OPEN_EXR'
        gtflow_out.base_path = res_paths['gtflow']

    # create node for saving segmentation
    if(params['output_types']['segm']):
        segm_out = tree.nodes.new('CompositorNodeOutputFile')
        segm_out.location = 40, 400
        segm_out.format.file_format = 'OPEN_EXR'
        segm_out.base_path = res_paths['segm']

    # merge fg and bg images
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs['Image'], mix.inputs[2])

    if(params['output_types']['vblur']):
        tree.links.new(mix.outputs[0], vblur.inputs[0])                # apply vector blur on the bg+fg image,
        tree.links.new(layers.outputs['Z'], vblur.inputs[1])           #   using depth,
        tree.links.new(layers.outputs['Speed'], vblur.inputs[2])       #   and flow.
        tree.links.new(vblur.outputs[0], vblur_out.inputs[0])          # save vblurred output

    tree.links.new(mix.outputs[0], composite_out.inputs[0])            # bg+fg image
    if(params['output_types']['fg']):
        tree.links.new(layers.outputs['Image'], fg_out.inputs[0])      # save fg
    if(params['output_types']['depth']):
        tree.links.new(layers.outputs['Z'], depth_out.inputs[0])       # save depth
    if(params['output_types']['normal']):
        tree.links.new(layers.outputs['Normal'], normal_out.inputs[0]) # save normal
    if(params['output_types']['gtflow']):
        tree.links.new(layers.outputs['Speed'], gtflow_out.inputs[0])  # save ground truth flow
    if(params['output_types']['segm']):
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])  # save segmentation

# creation of the spherical harmonics material, using an OSL script
def create_sh_material(tree, sh_path, img=None):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = sh_path #'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def init_scene(scene, params, gender='female'):
    # load fbx model
    # note that fbx model contains blocks_keys.
    bpy.ops.import_scene.fbx(filepath=join(params['smpl_data_folder'], 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
                             axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '%s_avg' % gender[0]
    ob = bpy.data.objects[obname]

    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    # set active camera.
    scn = bpy.context.scene
    scn.objects.active = cam_ob
    cam_ob.matrix_world = Matrix(((0., 0., 1, 7.0),
                                 (0., -1, 0., -1.0),
                                 (-1., 0., 0., 0.),
                                 (0.0, 0.0, 0.0, 1.0)))
    cam_ob.data.angle = math.radians(40)
    cam_ob.data.lens =  60
    cam_ob.data.clip_start = 0.1 ## defaults end at 1000
    cam_ob.data.sensor_width = 32

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_material_index  = True

    # set render size
    scn.render.resolution_x = params['resy']
    scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    return(ob, obname, arm_ob, cam_ob)

# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    # rotation - eye matrix???
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    if frame is not None:
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    # overwrite the old frame data with the newest data.
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

def get_bone_locs(obname, arm_ob, scene, cam_ob):
    n_bones = 24
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))
    bone_locations_2d = np.empty((n_bones, 2))
    bone_locations_3d = np.empty((n_bones, 3), dtype='float32')

    # obtain the coordinates of each bone head in image space
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.head)
        co_3d = arm_ob.matrix_world * bone.head
        bone_locations_3d[ibone] = (co_3d.x,
                                 co_3d.y,
                                 co_3d.z)
        bone_locations_2d[ibone] = (round(co_2d.x * render_size[0]),
                                 round(co_2d.y * render_size[1]))
    return(bone_locations_2d, bone_locations_3d)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, cam_ob)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)

# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)

    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]


    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}
            #  print({k:v.shape for k,v in cmu_parms[seq.replace('pose_', '')].items()})

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    #  print(fshapes.shape)

    return(cmu_parms, fshapes, name)

import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))


def print_resource_info(bg_path, smpl_data_folder, smpl_data_filename, verbose=True):
    ## load all bg images
    log_message("Listing background images")
    nh_txt_paths_dict = {}
    for split in ["train", "test"]:
        bg_names = join(bg_path, '%s_img.txt' % split)
        nh_txt_paths = []
        with open(bg_names) as f:
            for line in f:
                nh_txt_paths.append(join(bg_path, line))
        nh_txt_paths_dict[split] = nh_txt_paths

    ## grab clothing names
    log_message("Listing clothes images")
    clothing_option = "nongrey"
    log_message("clothing: %s" % clothing_option)
    txt_paths_dict = {}
    for gender in ["female", "male"]:
        for split in ["train", "test"]:
            with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % (gender, split) ) ) as f:
                txt_paths = f.read().splitlines()
                txt_paths = [k for k in txt_paths if clothing_option in k]
            txt_paths_dict[gender + "_" + split] = txt_paths

    ## load smpl motion data
    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    n_min_frames, n_max_frames = 1e8, 0
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            data = smpl_data[seq]
            if n_min_frames > len(data):
                n_min_frames = len(data)
            if n_max_frames < len(data):
                n_max_frames = len(data)

    if verbose:
        print("#seq: ", len(smpl_data.files))
        print("#frame: (", n_min_frames, ",", n_max_frames, ")")
        print("#female_shape: ", smpl_data["femaleshapes"].shape)
        print("#male_shape: ", smpl_data["maleshapes"].shape)
        for k in txt_paths_dict:
            print("cloth: " + k + ": ", len(txt_paths_dict[k]))
        for k in nh_txt_paths_dict:
            print("bg: " + split + ": ", len(nh_txt_paths_dict[k]))

    return nh_txt_paths_dict, txt_paths_dict, len(smpl_data.files), \
max(smpl_data["femaleshapes"].shape[0],smpl_data["maleshapes"].shape[0])

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def set_salient(salient):
    if salient:
        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        set_salient.oldbuffer = old
    else:
        if hasattr(set_salient, "oldbuffer"):
            old = set_salient.oldbuffer # it doesn't exist yet, so initialize it
            # disable output redirection
            os.close(1)
            os.dup(old)
            os.close(old)


def gen_rand_humans(setting):
    ## setup basic variables
    split = setting.split
    gender = setting.gender
    idx_shape = setting.ishape
    idx_cloth = setting.icloth
    idx_bg = setting.ibg

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_root = params['tmp_path']
    out_path = params['output_path']
    stepsize = params['stepsize']

    # check if already computed
    identifier = "_".join(["image", split, gender, "%04d"%idx_shape])
    tmp_path = join(tmp_root, identifier)
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)
    if not exists(tmp_path):
        mkdir_safe(tmp_path)
    res_paths = {k:join(tmp_root, identifier.replace("image", k) )
                 for k in params['output_types'] if params['output_types'][k]}

    # create copy-spher.harm. directory if not exists
    sh_dir = join(tmp_path, 'spher_harm')
    if not exists(sh_dir):
        mkdir_safe(sh_dir)
    sh_dst = join(sh_dir, 'sh.osl')
    os.system('cp spher_harm/sh.osl %s' % sh_dst)

    ### setup blender
    log_message("Setup blender")
    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    ## load all bg images
    log_message("Listing background images")
    bg_names = join(bg_path, '%s_img.txt' % split)
    nh_txt_paths = []
    with open(bg_names) as f:
        for line in f:
            nh_txt_paths.append(join(bg_path, line))

    ## grab clothing names
    clothing_option = "nongrey"
    log_message("clothing: %s" % clothing_option)
    with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % (gender, split) ) ) as f:
        txt_paths = f.read().splitlines()
        txt_paths = [k for k in txt_paths if clothing_option in k]


    ## TODO: random
    # random clothing texture
    idx_cloth = idx_cloth%len(txt_paths)
    cloth_img_name = txt_paths[idx_cloth]
    cloth_img_name = join(smpl_data_folder, cloth_img_name)
    cloth_img = bpy.data.images.load(cloth_img_name)

    # random background
    idx_bg = idx_bg%len(nh_txt_paths)
    bg_img_name = nh_txt_paths[idx_bg][:-1]
    bg_img = bpy.data.images.load(bg_img_name)

    ## load smpl motion data
    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

    ## start to setup the scene
    log_message("Building materials tree")
    mat_tree = bpy.data.materials['Material'].node_tree
    create_sh_material(mat_tree, sh_dst, cloth_img)
    create_composite_nodes(scene.node_tree, params, res_paths,
                           img=bg_img)

    log_message("Initializing scene")
    set_salient(1)
    ob, obname, arm_ob, cam_ob = init_scene(scene, params, gender)
    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob
    segmented_materials = True #True: 0-24, False: expected to have 0-1 bg/fg
    set_salient(0)

    # create material segmentation
    log_message("Creating materials segmentation")
    if segmented_materials:
        materials = create_segmentation(ob, params)
    else:
        materials = {'FullBody': bpy.data.materials['Material']}

    # load shape, pose for the given sequence index.
    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname,
                                              idx=idx_seq, gender=gender)

    # pick random real body shape
    log_message("Loaded body data for %s" % name)
    if split == 'train':
        fshapes = fshapes[:int(len(fshapes)*0.6)]
    elif split == 'val':
        fshapes = fshapes[int(len(fshapes)*0.6):int(len(fshapes)*0.8)]
    elif split == 'test':
        fshapes = fshapes[int(len(fshapes)*0.8):]
    shape = fshapes[idx_shape%(len(fshapes))] #+random_shape(.5) can add noise

    ## setup the pelvis to origin for ease
    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

    # create output directory
    if not exists(out_path):
        mkdir_safe(out_path)

    # spherical harmonics material needs a script to be loaded and compiled
    scs = []
    for mname, material in materials.items():
        scs.append(material.node_tree.nodes['Script'])
        scs[-1].filepath = sh_dst
        scs[-1].update()

    data = cmu_parms[name]
    fbegin = 0
    fend = len(data['poses'])//stepsize*min(stepsize, 150)

    log_message("Computing how many frames to allocate")
    N = len(data['poses'][fbegin:fend:stepsize])
    log_message("Allocating %d frames in mat file" % N)

    ## clearup, and put the origin at zeros
    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy())\
            - Vector((-1., 1., 1.))
    orig_cam_loc = cam_ob.location.copy()
    reset_joint_positions(Vector((0,0,0)), shape, ob, arm_ob, obname, scene,
                          cam_ob, smpl_data['regression_verts'],
                          smpl_data['joint_regressor'])
    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()


    ### lighting and rendering parameters
    scene.node_tree.nodes['Image'].image = bg_img
    ## lighting random:
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[0] = .5 + .9 * np.random.rand()
    sh_coeffs[1] = -.7 * np.random.rand()
    # TODO: what's the function??
    for part, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)
    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish+1].default_value = coeff
    get_real_frame = lambda ifr: ifr

    # iterate over the keyframes and render
    # LOOP TO RENDER
    render_path = res_paths["image"]
    for seq_frame, (pose, trans) in enumerate(
        zip(data['poses'][fbegin:fend:stepsize],
            data['trans'][fbegin:fend:stepsize])):

        # disable render output
        set_salient(1)
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        # shape at the origin, and translate the camera inversly
        apply_trans_pose_shape(Vector((0, 0, 0)), pose, shape, ob, arm_ob,
                               obname, scene, cam_ob, get_real_frame(seq_frame))
        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, 0), 'XYZ'))
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        scene.update()

        cam_ob.location = orig_cam_loc.copy() - arm_ob.matrix_world.copy()*Vector(trans)
        cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))

#          if seq_frame == 0:
            #  # Bodies centered only in each minibatch of clipsize frames
            #  new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            #  cam_ob.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
            #  #  cam_ob.location = orig_cam_loc.copy() - Vector(trans)
            #  cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))


        # Render the view from all given cameras
        log_message("Rendering frame %d" % seq_frame)
        scene.render.use_antialiasing = False
        res_paths["image"] = render_path + "_%04d.png"%get_real_frame(seq_frame)
        scene.render.filepath = res_paths["image"]
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        set_salient(0)

        # each frame data as a mat file
        ## save key information (bg, cloth, light, shape, pose, vblur, gender, split)
        dict_info = {}
        dict_info['bg'] = bg_img_name
        dict_info['cloth'] = cloth_img_name
        dict_info['light'] = sh_coeffs
        dict_info['cam_loc'] = np.array(cam_ob.location)
        dict_info['pose'] = pose
        dict_info['shape'] = shape
        dict_info['split'] = split
        dict_info['gender'] = gender
        dict_info['seqs'] = np.array([idx_seq, idx_shape, stepsize], np.int32)
        dict_info['vblur'] = params["vblur"]
        dict_info['camera'] = np.array(get_calibration_matrix_K_from_blender(
            cam_ob.data))
        dict_info['camera_R'] = np.zeros((3,))
        dict_info['camera_T'] = np.array(-cam_ob.location)
        # NOTE:
        # ideally, pixels should be readable from a viewer node, but I get only zeros
        # --> https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/
        # len(np.asarray(bpy.data.images['Render Result'].pixels) is 0
        # Therefore we write them to temporary files and read with OpenEXR library (available for python2) in main_part2.py
        # Alternatively, if you don't want to use OpenEXR library, the following commented code does loading with Blender functions, but it can cause memory leak.
        # If you want to use it, copy necessary lines from main_part2.py such as definitions of dict_normal, matfile_normal...
        ## TODO: where to store???
        ## list(image, depth, seg, normal, gt_flow, vblur, fg)
        imagefile_path = join(out_path, "image/" + identifier + "_%04d.png"%
                              get_real_frame(seq_frame))
        for k, folder in res_paths.items():
          if not k== 'vblur' and not k=='fg':
             filename = "Image%04d.exr"%get_real_frame(seq_frame)
             if k == "image":
                 render_img = bpy.data.images.load(folder)
             else:
                 render_img = bpy.data.images.load(folder + "/" + filename)
             # render_img.pixels size is width * height * 4 (rgba)
             arr = np.array(render_img.pixels[:]).reshape(
                 (params["resx"], params["resy"], -1))[::-1, :, :] # images are vertically flipped
             print(k, arr.shape)
             if k == 'normal':# 3 channels, original order
                 mat = arr[:,:, :3]
             elif k == 'gtflow':
                 mat = arr[:,:, 1:3]
             elif k == 'depth':
                 mat = arr[:,:, 0]
                 mat_c = mat.copy()
                 print("depths: ")
                 print(np.unique(mat_c.flatten()))
                 depth_m = sorted(mat_c.flatten())[-2]
                 mat_c[mat_c > depth_m] = depth_m + 1.0
                 cv2.imwrite(imagefile_path.replace("image", "depth"),
                             255.0/(mat_c+1e-5))
             elif k == 'segm':
                 mat = arr[:,:,0]
             elif k == "image":
                 mat = arr[:,:,:3]
                 cv2.imwrite(imagefile_path, mat)
             if(k in params['output_types']):
                 dict_info[k] = mat
             # remove the image to release memory, object handles, etc.
             render_img.user_clear()
             bpy.data.images.remove(render_img)
        dict_info['image'] = cv2.imread(scene.render.filepath)
        # bone locations should be saved after rendering so that the bones are updated
        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
        dict_info['joints2D'] = np.transpose(bone_locs_2D)
        dict_info['joints3D'] = np.transpose(bone_locs_3D)
        ## export rendering, parameters, and meshes to the output directory
        log_message("Export parameters, images, and meshes for " + imagefile_path)
        dd.io.save(imagefile_path.replace("image", "mat") , dict_info)
        # export human body
        bpy.ops.object.select_all(action='DESELECT')
        ob.select = True
        bpy.ops.export_scene.obj(filepath=imagefile_path.replace("image", "mesh").replace(".png", ".obj"),
                                 use_selection=True, use_uvs=False, use_materials=False)
        bpy.ops.object.select_all(action='DESELECT')

        # TODO: ????
        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))
    return fend//stepsize + 1

def gen_multi_seqs():
    global start_time
    start_time = time.time()
    log_message("Remove out/, and tmp/")
    if os.path.exists("../out/"):
        shutil.rmtree("../out/")
    if os.path.exists("../tmp/"):
        shutil.rmtree("../tmp/")

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']

    print("Resource information: ")
    bg_splits, cloth_splits, nseqs, nshapes = print_resource_info(
        bg_path, smpl_data_folder, smpl_data_filename)
    for gender in ["female", "male"]:
        for split in ["train", "test"]:
            for irun in range(500):
                nbg = len(bg_splits[split])
                ncloth = len(cloth_splits[gender + "_" + split])
                # setup params
                setting = edict()
                setting.split = split ##
                setting.gender = gender ## gender, 0/1
                setting.iseq = random.randint(0, nseqs) ## seq id, rand
                setting.ibg = random.randint(0, nbg) ## bg image, rand
                setting.ishape = random.randint(0, nshapes) ## shape idx, rand
                setting.icloth = random.randint(0, ncloth) ## cloth idx, rand
                setting.irun = irun
                print("Generate a sequence with settings: ", setting)
                nframes = gen_one_seqs(setting)
                print("#%d frames are generated."%(nframes))


if __name__ == '__main__':
    gen_multi_seqs()

