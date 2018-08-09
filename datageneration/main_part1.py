#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : datageneration/main_part1.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 03.08.2018
# Last Modified Date: 06.08.2018
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import sys
import os
import random
import math
import bpy
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from glob import glob
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam
from easydict import EasyDict as edict
import pickle
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
    res_paths = {k:join(params['tmp_path'], '%05d_%s'%(idx_run, k)) for k in params['output_types'] if params['output_types'][k]}
    # vblur
    # depth
    # normal
    # segm
    for k in res_paths:
        path_k = res_paths[k] + "/" + "Image%04d.exr"%idx_frame
        if os.path.exists(path_k):
            os.rename(path_k, path_k.replace(".exr", "%s.exr"%suffix))
        elif os.path.exists(path_k.repalce(".exr", ".png")):
            os.rename(path_k, path_k.replace(".exr", "%s.png"%suffix))


# create the different passes that we render
def create_composite_nodes(tree, params, img=None, idx=0):
    res_paths = {k:join(params['tmp_path'], '%05d_%s'%(idx, k)) for k in params['output_types'] if params['output_types'][k]}

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
        vblur.factor = params['vblur_factor']
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

    return(res_paths)

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
    print(len(uv_im.outputs), len(script.inputs) )
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

    camera_params = params["camera_params"]
    for ii in range(camera_params["camera_num"]):
        if ii>=1:
            cam = bpy.data.cameras.new(name="camera_%d"%ii)
            cam_ob = bpy.data.objects.new("camera_%d"%ii, cam)
        else:
            cam_ob_ret = cam_ob

        cam_ob.lock_rotations_4d = False
        scn.objects.active = cam_ob

        loc = camera_params["camera_loc"][ii]
        cam_ob.matrix_world = Matrix(((0., 0., 1, loc[0]),
                                 (0., -1, 0., loc[1]),
                                 (-1., 0., 0., loc[2]),
                                 (0.0, 0.0, 0.0, 1.0)))
        loc, rot = cam_ob.matrix_world.decompose()[:2]
        print("camera info: ")
        print("translation: " + str(loc))
        print("rotation: " + str(cam_ob.rotation_euler))
        cam_ob.data.angle = math.radians(camera_params["camera_fov"][ii])
        cam_ob.data.lens =  camera_params["camera_len"][ii]
        cam_ob.data.clip_start = 0.1
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

    return(ob, obname, arm_ob, cam_ob_ret)

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


    ## verbose
    #  print("******************************")
    #  print("pose&shape: ")
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


def main_part1(file_cfg):
    # time logging
    global start_time
    start_time = time.time()
    with open(file_cfg, "rb") as f:
        print("load configs from: " + file_cfg)
        config = pickle.load(f)

    seed_number = config.seed_number
    idx_gender = config.idx_gender
    idx_bg = config.idx_bg
    idx_fshape = config.idx_fshape
    idx_cloth = config.idx_cloth
    disp_bg = config.disp_bg

    # human data source:
    idx = config.idx_seq
    ishape = config.idx_ishape
    stride = config.stride

    # human z-rotation
    random_zrot = config.zrot

    # this part is fixed for different runs.
    # motion blur
    vblur_factor = np.random.normal(0.5, 0.5)

    # lighting random:
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[0] = .5 + .9 * np.random.rand()
    sh_coeffs[1] = -.7 * np.random.rand()

    # camera is setup with configs in confg, thus can only be random for every run.

    # random parameters:
    random.seed(seed_number)
    np.random.seed(seed_number)

    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)

    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50

    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    (runpass, idx) = divmod(idx, len(idx_info))

    log_message("runpass: %d" % runpass)
    log_message("output idx: %d" % idx)
    idx_info = idx_info[idx]
    log_message("sequence: %s" % idx_info['name'])
    log_message("nb_frames: %f" % idx_info['nb_frames'])
    log_message("use_split: %s" % idx_info['use_split'])

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    cam_params = config.load_file("config", 'CAMERA_PARAMS')
    params["camera_params"] = cam_params
    print("cameras: ")
    print(cam_params)

    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']

    # compute number of cuts
    nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
    log_message("Max ishape: %d" % (nb_ishape - 1))

    if ishape == None:
        exit(1)
    ishape = ishape%nb_ishape
    assert(ishape < nb_ishape)

    # name is set given idx
    name = idx_info['name']
    output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    params['output_path'] = output_path
    tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    params['tmp_path'] = tmp_path

    # check if already computed
    #  + clean up existing tmp folders if any
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)
    rgb_vid_filename = "%s_c%04d.mp4" % (join(output_path, name.replace(' ', '')), (ishape + 1))
    #if os.path.isfile(rgb_vid_filename):
    #    log_message("ALREADY COMPUTED - existing: %s" % rgb_vid_filename)
    #    return 0

    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)

    # initialize RNG with seeds from sequence id
    import hashlib
    s = "synth_data:%d:%d:%d" % (idx, runpass,ishape)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))




    # vblure ???
    if(output_types['vblur']):
        params['vblur_factor'] = vblur_factor

    log_message("Setup Blender")

    # create copy-spher.harm. directory if not exists
    sh_dir = join(tmp_path, 'spher_harm')
    if not exists(sh_dir):
        mkdir_safe(sh_dir)
    sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
    os.system('cp spher_harm/sh.osl %s' % sh_dst)

    genders = {0: 'female', 1: 'male'}
    # pick random gender
    idx_gender = idx_gender%2
    gender = genders[idx_gender]

    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    log_message("Listing background images")
    bg_names = join(bg_path, '%s_img.txt' % idx_info['use_split'])
    nh_txt_paths = []
    with open(bg_names) as f:
        for line in f:
            nh_txt_paths.append(join(bg_path, line))

    # grab clothing names
    log_message("clothing: %s" % clothing_option)
    with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % ( gender, idx_info['use_split'] ) ) ) as f:
        txt_paths = f.read().splitlines()

    # if using only one source of clothing
    if clothing_option == 'nongrey':
        txt_paths = [k for k in txt_paths if 'nongrey' in k]
    elif clothing_option == 'grey':
        txt_paths = [k for k in txt_paths if 'nongrey' not in k]

    # random clothing texture
    idx_cloth = idx_cloth%len(txt_paths)
    cloth_img_name = txt_paths[idx_cloth]
    cloth_img_name = join(smpl_data_folder, cloth_img_name)
    cloth_img = bpy.data.images.load(cloth_img_name)

    # random background
    idx_bg = idx_bg%len(nh_txt_paths)
    bg_img_name = nh_txt_paths[idx_bg][:-1]
    bg_img = bpy.data.images.load(bg_img_name)

    log_message("Building materials tree")
    mat_tree = bpy.data.materials['Material'].node_tree
    create_sh_material(mat_tree, sh_dst, cloth_img)
    res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=idx)

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

    log_message("Initializing scene")
    ob, obname, arm_ob, cam_ob = init_scene(scene, params, gender)

    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob
    segmented_materials = True #True: 0-24, False: expected to have 0-1 bg/fg

    log_message("Creating materials segmentation")
    # create material segmentation
    if segmented_materials:
        materials = create_segmentation(ob, params)
        prob_dressed = {'leftLeg':.5, 'leftArm':.9, 'leftHandIndex1':.01,
                        'rightShoulder':.8, 'rightHand':.01, 'neck':.01,
                        'rightToeBase':.9, 'leftShoulder':.8, 'leftToeBase':.9,
                        'rightForeArm':.5, 'leftHand':.01, 'spine':.9,
                        'leftFoot':.9, 'leftUpLeg':.9, 'rightUpLeg':.9,
                        'rightFoot':.9, 'head':.01, 'leftForeArm':.5,
                        'rightArm':.5, 'spine1':.9, 'hips':.9,
                        'rightHandIndex1':.01, 'spine2':.9, 'rightLeg':.5}
    else:
        materials = {'FullBody': bpy.data.materials['Material']}
        prob_dressed = {'FullBody': .6}

    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
    orig_cam_loc = cam_ob.location.copy()

    #TODO: EMPTY????
    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)


    log_message("Loaded body data for %s" % name)

    nb_fshapes = len(fshapes)
    if idx_info['use_split'] == 'train':
        fshapes = fshapes[:int(nb_fshapes*0.8)]
    elif idx_info['use_split'] == 'test':
        fshapes = fshapes[int(nb_fshapes*0.8):]

    # pick random real body shape
    idx_fshape = idx_fshape%len(fshapes)
    shape = fshapes[idx_fshape] #+random_shape(.5) can add noise
    ndofs = 10

    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

    # create output directory
    if not exists(output_path):
        mkdir_safe(output_path)

    # spherical harmonics material needs a script to be loaded and compiled
    scs = []
    for mname, material in materials.items():
        scs.append(material.node_tree.nodes['Script'])
        scs[-1].filepath = sh_dst
        scs[-1].update()

    rgb_dirname = name.replace(" ", "") + '_c%04d.mp4' % (ishape + 1)
    rgb_path = join(tmp_path, rgb_dirname)

    data = cmu_parms[name]

    nsample = len(data['poses'])
    fbegin = (ishape*stepsize*stride)%nsample
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    fbegin = fend - stepsize*clipsize

    log_message("Computing how many frames to allocate")
    N = len(data['poses'][fbegin:fend:stepsize])
    log_message("Allocating %d frames in mat file" % N)

    # force recomputation of joint angles unless shape is all zeros
    curr_shape = np.zeros_like(shape)
    nframes = len(data['poses'][::stepsize])

    matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
    log_message('Working on %s' % matfile_info)

    # allocate
    # save the basic information.
    dict_info = {}
    dict_info['bg'] = np.zeros((N,), dtype=np.object) # background image path
    dict_info['camLoc'] = np.empty(3) # (1, 3)
    dict_info['clipNo'] = ishape +1
    dict_info['cloth'] = np.zeros((N,), dtype=np.object) # clothing texture image path
    dict_info['gender'] = np.empty(N, dtype='uint8') # 0 for male, 1 for female
    dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32') # 2D joint positions in pixel space
    dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32') # 3D joint positions in world coordinates
    dict_info['light'] = np.empty((9, N), dtype='float32')
    dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32') # joint angles from SMPL (CMU)
    dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
    dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
    dict_info['zrot'] = np.empty(N, dtype='float32')
    dict_info['stride'] = stride

    if name.replace(" ", "").startswith('h36m'):
        dict_info['source'] = 'h36m'
    else:
        dict_info['source'] = 'cmu'

    if(output_types['vblur']):
        dict_info['vblur_factor'] = np.empty(N, dtype='float32')

    # for each clipsize'th frame in the sequence
    get_real_frame = lambda ifr: ifr
    batch_it = 0
    curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                       cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])

    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    # create a keyframe animation with pose, translation, blendshapes and camera motion
    # LOOP TO CREATE 3D ANIMATION
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))
        dict_info['shape'][:, iframe] = shape[:ndofs]
        dict_info['pose'][:, iframe] = pose
        dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
        if(output_types['vblur']):
            dict_info['vblur_factor'][iframe] = vblur_factor

        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        dict_info['zrot'][iframe] = random_zrot

        scene.update()

        # Bodies centered only in each minibatch of clipsize frames
        if seq_frame == 0:
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            # setup location for each frame
            idx_cam = 0
            for obj in bpy.data.objects:
                if obj.type == "CAMERA":
                    v_trans = Vector((0, 0, 0))
                    if obj != cam_ob:
                        vec_cam = cam_params["camera_loc"]
                        for i in range(3):
                            v_trans[i] = vec_cam[idx_cam][i] - vec_cam[0][i]
                    obj.location = v_trans + orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
                    obj.keyframe_insert('location', frame=get_real_frame(seq_frame))
                    idx_cam += 1
                    if obj == cam_ob:
                        dict_info['camLoc'] = np.array(cam_ob.location)

    scene.node_tree.nodes['Image'].image = bg_img
    vsize = bg_img.size

    # add croped images;
    id_cam = 0
    img_list = []
    for ii,ob in enumerate(bpy.data.objects):
        if ob.type != "CAMERA":
            continue
        x_start = disp_bg*ii
        img_list.append(crop_image(bg_img, x_start, vsize[0], 0, vsize[1]))
        id_cam += 1


    # TODO: what's the function??
    for part, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish+1].default_value = coeff

    # export the scene with animations:
#      scene.collada_export("../out/human.dae", selected=True, include_children=True,
            #  include_armatures=True, use_blender_profile=True, include_shapekeys=True,
            #  include_uv_textures=True, include_material_textures=True, use_texture_copies=True)
    # bpy.ops.export_scene.fbx(filepath="../out/human.fbx", use_selection=True, embed_textures=True,
    #        bake_space_transform=True)

    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        dict_info['bg'][iframe] = bg_img_name
        dict_info['cloth'][iframe] = cloth_img_name
        dict_info['light'][:, iframe] = sh_coeffs

        scene.render.use_antialiasing = False

        log_message("Rendering frame %d" % seq_frame)

        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        idx_cam = 0
        for ii,ob in enumerate(bpy.data.objects):
            if ob.type != "CAMERA":
                continue
            v_trans = Vector((0, 0, 0))
            if obj != cam_ob:
                vec_cam = cam_params["camera_loc"]
                for i in range(3):
                    v_trans[i] = vec_cam[idx_cam][i] - vec_cam[0][i]
            scene.node_tree.nodes['Image'].image = img_list[idx_cam]
            scene.camera = ob
            scene.render.filepath = join(rgb_path, 'Image%04d_%01d.png' % (get_real_frame(seq_frame), ii))
            bpy.ops.render.render(write_still=True)
            # move files
            mv_result_files(params, idx_run=idx,
                    idx_frame=seq_frame, suffix="_%d"%idx_cam)
            idx_cam += 1

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

        # NOTE:
        # ideally, pixels should be readable from a viewer node, but I get only zeros
        # --> https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/
        # len(np.asarray(bpy.data.images['Render Result'].pixels) is 0
        # Therefore we write them to temporary files and read with OpenEXR library (available for python2) in main_part2.py
        # Alternatively, if you don't want to use OpenEXR library, the following commented code does loading with Blender functions, but it can cause memory leak.
        # If you want to use it, copy necessary lines from main_part2.py such as definitions of dict_normal, matfile_normal...

        #for k, folder in res_paths.items():
        #   if not k== 'vblur' and not k=='fg':
        #       path = join(folder, 'Image%04d.exr' % get_real_frame(seq_frame))
        #       render_img = bpy.data.images.load(path)
        #       # render_img.pixels size is width * height * 4 (rgba)
        #       arr = np.array(render_img.pixels[:]).reshape(resx, resy, 4)[::-1,:, :] # images are vertically flipped
        #       if k == 'normal':# 3 channels, original order
        #           mat = arr[:,:, :3]
        #           dict_normal['normal_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'gtflow':
        #           mat = arr[:,:, 1:3]
        #           dict_gtflow['gtflow_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'depth':
        #           mat = arr[:,:, 0]
        #           dict_depth['depth_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'segm':
        #           mat = arr[:,:,0]
        #           dict_segm['segm_%d' % (iframe + 1)] = mat.astype(np.uint8, copy=False)
        #
        #       # remove the image to release memory, object handles, etc.
        #       render_img.user_clear()
        #       bpy.data.images.remove(render_img)

        # bone locations should be saved after rendering so that the bones are updated
        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
        dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
        dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)

        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

    # save a .blend file for debugging:
    # bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path, 'pre.blend'))

    # save RGB data with ffmpeg (if you don't have h264 codec, you can replace with another one and control the quality with something like -q:v 3)
    cmd_ffmpeg_right = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p ''%s_c%04d_1.mp4''' % (join(rgb_path, 'Image%04d_1.png'), join(output_path, name.replace(' ', '')), (ishape + 1))
    cmd_ffmpeg_left = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p ''%s_c%04d_2.mp4''' % (join(rgb_path, 'Image%04d_2.png'), join(output_path, name.replace(' ', '')), (ishape + 1))
    os.system(cmd_ffmpeg_left)
    os.system(cmd_ffmpeg_right)
    log_message("Generating RGB video (%s)" % cmd_ffmpeg_left)

#      if(output_types['vblur']):
    #      cmd_ffmpeg_vblur = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ''%s_c%04d.mp4''' % (join(res_paths['vblur'], 'Image%04d_%01d.png'), join(output_path, name.replace(' ', '')+'_vblur'), (ishape + 1))
    #      log_message("Generating vblur video (%s)" % cmd_ffmpeg_vblur)
    #      os.system(cmd_ffmpeg_vblur)
    #
    #  if(output_types['fg']):
    #      cmd_ffmpeg_fg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s_c%04d.mp4''' % (join(res_paths['fg'], 'Image%04d_%01d.png'), join(output_path, name.replace(' ', '')+'_fg'), (ishape + 1))
    #      log_message("Generating fg video (%s)" % cmd_ffmpeg_fg)
        #  os.system(cmd_ffmpeg_fg)

    # compress the corresponding file.
    cmd_tar = 'tar -czf %s/%s.tar.gz -C %s %s' % (output_path, rgb_dirname, tmp_path, rgb_dirname)
    log_message("Tarballing the images (%s)" % cmd_tar)
    os.system(cmd_tar)
    print("tar done.")

    # save annotation excluding png/exr data to _info.mat file
    import scipy.io
    scipy.io.savemat(matfile_info, dict_info, do_compression=True)
    print("save annotation done.")


def main():
    import argparse
    print("args: ")
    print(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')
    parser.add_argument('--file', type=str, default='',
                        help='file name of params configuration.')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


    if args.file == "":
        # setup params
        config = edict()
        config.idx_seq = args.idx
        config.idx_ishape = args.ishape
        config.stride = args.stride

        # this part will be changed for different runs.
        config.seed_number = 100
        config.idx_gender = 0
        config.idx_bg = 0
        config.idx_fshape = 0
        config.idx_cloth = 0
        config.disp_bg = random.randint(1, 5)

        # human z-rotation
        config.zrot = 2*np.pi*np.random.rand()
        file_cfg = "params.cfg"
        with open(file_cfg, 'wb') as f:
            pickle.dump(config, f, protocol=2)
    else:
        # load param file:
        file_cfg = args.file

    main_part1(file_cfg)


if __name__ == '__main__':
    main()

