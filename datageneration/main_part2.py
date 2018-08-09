import sys
import os
from os import remove
from os.path import join, dirname, realpath, exists
import numpy as np
from easydict import EasyDict as edict
import pickle

def load_body_data(smpl_data, idx=0):
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
    return(cmu_parms, name)


import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))


def main_part2(file_cfg):
    # time logging
    global start_time
    start_time = time.time()

    # param list:
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


    from pickle import load
    import argparse
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

    log_message("start part 2")

    import hashlib
    import random
    # initialize random seeds with sequence id
    s = "synth_data:%d:%d:%d" % (idx, runpass, ishape)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)

    # import configuration
    import config
    params = config.load_file('config', 'SYNTH_DATA')

    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    resy = params['resy']
    resx = params['resx']
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    openexr_py2_path = params['openexr_py2_path']

    name = idx_info[idx]["name"]

    # compute number of cuts
    nb_ishape = max(1, int(np.ceil((idx_info[idx]['nb_frames'] - (clipsize - stride))/stride)))
    ishape = ishape%nb_ishape


    output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    log_message("output path: " + output_path)
    tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))

    # check whether openexr_py2_path is loaded from configuration file
    if 'openexr_py2_path' in locals() or 'openexr_py2_path' in globals():
        for exr_path in openexr_py2_path.split(':'):
            sys.path.insert(1, exr_path)

    # to read exr imgs
    import OpenEXR
    import array
    import Imath

    log_message("Loading SMPL data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    cmu_parms, name = load_body_data(smpl_data, idx)

    res_paths = {k:join(tmp_path, '%05d_%s'%(idx, k)) for k in output_types if output_types[k]}

    data = cmu_parms[name]
    nframes = len(data['poses'][::stepsize])

    # .mat files
    matfile_normal = join(output_path, name.replace(" ", "") + "_c%04d_normal.mat" % (ishape + 1))
    matfile_gtflow = join(output_path, name.replace(" ", "") + "_c%04d_gtflow.mat" % (ishape + 1))
    matfile_depth = join(output_path, name.replace(" ", "") + "_c%04d_depth.mat" % (ishape + 1))
    matfile_segm = join(output_path, name.replace(" ", "") + "_c%04d_segm.mat" % (ishape + 1))
    dict_normal = {}
    dict_gtflow = {}
    dict_depth = {}
    dict_segm = {}
    get_real_frame = lambda ifr: ifr
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # overlap determined by stride (# subsampled frames to skip)
    fbegin = ishape*stepsize*stride
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    # LOOP OVER FRAMES
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame

        log_message("Processing frame %d" % iframe)

        for k, folder in res_paths.items():
            if not k== 'vblur' and not k=='fg':
                for ii in range(2):
                    path = join(folder, 'Image%04d_%d.exr' % (get_real_frame(seq_frame), ii))
                    exr_file = OpenEXR.InputFile(path)
                    if k == 'normal':
                        mat = np.transpose(np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")], (3, resx, resy)), (1, 2, 0))
                        dict_normal['normal_%04d_%01d' % (iframe + 1, ii+1)] = mat.astype(np.float32, copy=False) # +1 for the 1-indexing
                    elif k == 'gtflow':
                        mat = np.transpose(np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G")], (2, resx, resy)), (1, 2, 0))
                        dict_gtflow['gtflow_%04d_%01d' % (iframe + 1, ii+1)] = mat.astype(np.float32, copy=False)
                    elif k == 'depth':
                        mat = np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R")], (resx, resy))
                        dict_depth['depth_%04d_%01d' % (iframe + 1, ii+1)] = mat.astype(np.float32, copy=False)
                    elif k == 'segm':
                        mat = np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R")], (resx, resy))
                        dict_segm['segm_%04dd_%01d' % (iframe + 1, ii+1)] = mat.astype(np.uint8, copy=False)
                    # remove(path)

    print("render infos: ")
    print(res_paths)
    print("#normal: %d"%(len(dict_normal.keys())))
    print("#depth: %d"%(len(dict_depth.keys())))
    print("#segm: %d"%(len(dict_segm.keys())))
    import scipy.io
    scipy.io.savemat(matfile_normal, dict_normal, do_compression=True)
    scipy.io.savemat(matfile_gtflow, dict_gtflow, do_compression=True)
    scipy.io.savemat(matfile_depth, dict_depth, do_compression=True)
    scipy.io.savemat(matfile_segm, dict_segm, do_compression=True)

    # cleaning up tmp
    if tmp_path != "" and tmp_path != "/":
        log_message("Cleaning up tmp")
        os.system('rm -rf %s' % tmp_path)

    log_message("Completed batch")

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

    args = parser.parse_args(sys.argv[1:])

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

        # human z-rotation
        file_cfg = "params.cfg"
        with open(file_cfg, 'wb') as f:
            pickle.dump(config, f)
    else:
        # load param file:
        file_cfg = args.file
    main_part2(file_cfg)


if __name__ == '__main__':
    main()


