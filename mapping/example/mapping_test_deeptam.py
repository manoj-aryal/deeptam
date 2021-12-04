import tensorflow as tf
from deeptam_mapper.models.helpers import *
from deeptam_mapper.utils.helpers import *
from deeptam_mapper.utils.datatypes import *
import deeptam_mapper.evaluation.metrics as metrics
from deeptam_mapper.utils.vis_utils import convert_array_to_colorimg,convert_array_to_grayimg
import sys
import numpy as np
import time
from PIL import Image
import os 
from collections import namedtuple
DepthMetrics = namedtuple('DepthMetrics', ['l1_inverse', 'scale_invariant', 'abs_relative'])
import pickle
import matplotlib.pyplot as plt

def init_visualization(title):
    """Initializes a simple visualization for tracking
    
    title: str
    """
    fig = plt.figure()
    fig.set_size_inches(10.5, 8.5)
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(2,2,1)
    
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Keyframe')
           
    ax2 = fig.add_subplot(2,2,2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax2.set_title('Current frame: ')
    ax3 = fig.add_subplot(2,2,3)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    ax3.set_title('FB depth pr: frame ')
    ax4 = fig.add_subplot(2,2,4)
    
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title('NB depth pr: iter ')
    
    return [ax1, ax2, ax3, ax4]


def update_visualization(axes, 
                         image_cur,
                         image_cur_virtual,
                         depth_pr_fb,
                         depth_pr_nb,
                         title_suffixes):
    """ Updates the visualization for tracking
    
    axes: a list of plt.axes
    
    image_cur, image_cur_virtual: np.array
    
    depth_pr_fb, depth_pr_nb: np.array
    
    title_subfixes: a list of str
    
    """

    if image_cur_virtual is not None:
        image_cur = convert_array_to_colorimg(image_cur.squeeze())
        image_cur_virtual = convert_array_to_colorimg(image_cur_virtual.squeeze()) 
        depth_pr_vis_fb = convert_array_to_grayimg(depth_pr_fb.squeeze()) 
        if depth_pr_nb is not None:
            depth_pr_vis_nb = convert_array_to_grayimg(depth_pr_nb.squeeze()) 

        axes[0].imshow(np.array(image_cur))       
        axes[1].imshow(np.array(image_cur_virtual))
        axes[2].imshow(np.array(depth_pr_vis_fb))
        if depth_pr_nb is not None:
            axes[3].imshow(np.array(depth_pr_vis_nb))
            
        axes[0].set_title('Keyframe')
        axes[1].set_title('Current frame: ' + title_suffixes[1])
        axes[2].set_title('FB depth pr: frame '+ title_suffixes[2])
        axes[3].set_title('NB depth pr: iter ' + title_suffixes[3])

    plt.pause(0.5)

def compute_depth_metrics(pr, gt):
    """Computes depth errors
    pr: np.ndarray 
        The prediction as absolute depth values
    gt: np.ndarray
        The ground truth as absolute depth values
    """
    valid_mask = metrics.compute_valid_depth_mask(pr, gt)
    valid_pr = pr[valid_mask]
    valid_gt = gt[valid_mask]

    return DepthMetrics(**
        {
            'l1_inverse': metrics.l1_inverse(valid_pr, valid_gt),
            'scale_invariant': metrics.scale_invariant(valid_pr, valid_gt),
            'abs_relative': metrics.abs_relative(valid_pr, valid_gt),
        })


def create_cv_conf_from_sequence_py(depth_key,
                                    data_path,
                                    session,
                                    net,
                                    outputs):
    """Compute
    
    depth_key: np.array
    
    sub_seq_py: SeqPy
    
    session: tf.Session
    
    net:  mapping_mod.CVGenerateNetwork
    
    outputs: dict of Tensor
    """
    
    # frame_end = sub_seq_py.seq_len - 1
    cv_list = []
    cv_conf_list = []
    
    # image_key -> first image, curr_image (images), intrinsic (3x3 array in docs), rotation and translation (camera poses)
    # computes pairwise cost volume with its conf between the keyframe and all the other frames in the sequence
    key_image = Image.open(os.path.join(data_path, '1.png'))
    key_image = np.array(key_image)
    
    new_key_img = np.zeros((1,3, 240, 320))
    new_key_img[0, 0, :, :] = key_image[:240, :320] 

    new_key_img[0, 1, :, :] = key_image[:240, :320] 
    new_key_img[0, 2, :, :] = key_image[:240, :320]

    # new_key_img = new_key_img.astype(np.uint8)
    new_key_img = new_key_img/256
    print('************', np.amin(new_key_img),np.amax(new_key_img),new_key_img.dtype)

    intrinsics = [[0.6542779919, 0.86980780565, 0.50187017568, 0.51435659567]]
    intrinsics = np.array(intrinsics)
    translations = [[0.19685354621255852, -1.595713957578012, 0.73519086796852], [0.18154085392069225, -1.5849289029694344, 0.7287384958452199], [0.16108172350711408, -1.5726693667292535, 0.7202639745999071], [0.15035306443605384, -1.5673784821608066, 0.7157743826604706], [0.13496367315918617, -1.5572423176679897, 0.7088914225348102], [0.12517380388126392, -1.5510430313954267, 0.7043271417668118], [0.10447643467936002, -1.5387830059503473, 0.6947476258543928], [0.094395670536007, -1.5328477489208652, 0.6901238835331056], [0.06873287144397422, -1.51835913994019, 0.6775586225549253], [0.0535851160173455, -1.509147489741534, 0.6698895633838617]]
    rotations = [[ 0.04302139, -0.0434732 , -0.0449899 ], [ 0.03814083, -0.05114667, -0.04481178], [ 0.03787545, -0.06108375, -0.0441705 ], [ 0.03950699, -0.0666737 , -0.04218503],[ 0.03651359, -0.07335529, -0.0426389 ], [ 0.03601305, -0.07689447, -0.04570844], [ 0.03496616, -0.08587657, -0.04655286], [ 0.03490441, -0.08928628, -0.04752722],  [0.03526783, -0.09979656, -0.04992247],  [0.0336143 , -0.1043838 , -0.05126869]]

    for frame in range(1, 11):
        curr_image = Image.open(os.path.join(data_path, str(frame) + '.png'))
        curr_image = np.array(curr_image)
        curr_img_arr = np.zeros((1,3, 240, 320))
        curr_img_arr[0, 0, :, :] = curr_image[:240, :320] 

        curr_img_arr[0, 1, :, :] = curr_image[:240, :320] 
        curr_img_arr[0, 2, :, :] = curr_image[:240, :320]
        # curr_img_arr = curr_img_arr.astype(np.uint8)
        curr_img_arr = curr_img_arr/256
        print('curr_img', np.amax(curr_img_arr))
        curr_rotation = np.array(rotations[frame-1])
        curr_translation = np.array(translations[frame-1])
        feed_dict = {
                net.placeholders['depth_key']: depth_key,
                net.placeholders['image_key']: new_key_img,
                net.placeholders['image_current']:curr_img_arr,
                net.placeholders['intrinsics']:intrinsics,
                net.placeholders['rotation']:np.expand_dims(curr_rotation,axis=0),
                net.placeholders['translation']:np.expand_dims(curr_translation,axis=0),
        }
        # 1.0 0.5 0.5 1.1916667 0.005703583359718324 0.014408113434910774
        # print(feed_dict)
        cv_generate_out = session.run(outputs, feed_dict=feed_dict)
        
        cv = cv_generate_out['cv']
        print('*****cv*******',np.amin(cv),np.amax(cv))
        print('*****info cvvv*******','depth', np.amax(cv_generate_out['depth_label_tensor']),'warped', np.amax(cv_generate_out['warped']), 'mask', np.amax(cv_generate_out['mask']),
        'image_key', np.amax(cv_generate_out['image_key']), 'image_current', np.amax(cv_generate_out['image_current']), 'rotation', np.amax(cv_generate_out['rotation']), 'translation', np.amax(cv_generate_out['translation']), 'intrinsics', np.amax(cv_generate_out['intrinsics']), 'flows', np.amax(cv_generate_out['flows']), 'i', np.amax(cv_generate_out['i']), 'r', np.amax(cv_generate_out['r']), 't', np.amax(cv_generate_out['t']))
        cv_conf = cv_generate_out['cv_conf']
        print('*****cv-conf*******',np.amin(cv_conf),np.amax(cv_conf))
        depth_label_tensor = cv_generate_out['depth_label_tensor']
        cv_list.append(cv)
        cv_conf_list.append(cv_conf)
    
    # cost volume aggregation over all the frames in the sequence
    # cv_conf_list is all zero
    print('*****cv_list*******',np.amax(cv_conf_list),np.amax(cv_list), np.shape(cv_conf))
    cv_sum = np.zeros_like(cv)
    cv_conf_sum = np.zeros_like(cv_conf)
    for ind in range(len(cv_list)):
        cv_sum += cv_list[ind]*cv_conf_list[ind]
        cv_conf_sum += cv_conf_list[ind]
    print('cvvvv', np.amax(cv_sum), np.amax(cv_conf_sum), np.amin(cv_sum), np.amin(cv_conf_sum))
    cv_mean = cv_sum/cv_conf_sum
    print('shape', np.shape(cv_sum), np.shape(cv_conf_sum))
    print('cv_mean1', np.amax(cv_mean))
    cv_mean[cv_mean == np.inf] = 0
    print('cv_mean2', np.amax(cv_mean))
    cv_mean[cv_mean == -np.inf] = 0
    print('cv_mean3', np.amax(cv_mean))
    print('cv_mean3', np.amin(cv_mean))
    return np.nan_to_num(cv_mean), depth_label_tensor

def mapping_with_pose(
                        datafile,
                        mapping_mod_path,
                        checkpoints,
                        gpu_memory_fraction=None,
                        width = 320,
                        height = 240,
                        max_sequence_length=10,
                        nb_iterations_num=5,
                        savedir=None):
    
    tf.reset_default_graph()
    
    gpu_options = tf.GPUOptions()
    if not gpu_memory_fraction is None:
        gpu_options.per_process_gpu_memory_fraction=gpu_memory_fraction
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    
    # build mapping mod
    mapping_mod = load_myNetworks_module_noname(mapping_mod_path)

    # fixed band module
    fb_depth_net = mapping_mod.MappingFBNetwork(batch_size=1, width=width, height=height)
    fb_depth_outputs = fb_depth_net.build_net(**fb_depth_net.placeholders,state=fb_depth_net.placeholders_state)

    # narrow band module
    nb_depth_net = mapping_mod.MappingNBNetwork(batch_size=1, width=width, height=height)
    nb_depth_outputs = nb_depth_net.build_net(**nb_depth_net.placeholders,state=nb_depth_net.placeholders_state)

    # narrow band refinement module
    nb_refine_depth_net = mapping_mod.MappingNBRefineNetwork(batch_size=1, width=width, height=height)
    nb_refine_depth_outputs = nb_refine_depth_net.build_net(**nb_refine_depth_net.placeholders,state=nb_refine_depth_net.placeholders_state)

    # pairwise cost volume generation for fixed band module
    cv_fb_generate_net = mapping_mod.CVGenerateNetwork(batch_size=1, width=width, height=height, depth_scale_array=np.linspace(0.01,2.5,32))
    cv_fb_generate_outputs = cv_fb_generate_net.build_net(**cv_fb_generate_net.placeholders,state=cv_fb_generate_net.placeholders_state)

    # pairwise cost volume generation for narrow band module
    cv_nb_generate_net = mapping_mod.CVGenerateNetwork(batch_size=1, width=width, height=height, depth_scale_array=np.linspace(0.8,1.2,32))
    cv_nb_generate_outputs = cv_nb_generate_net.build_net(**cv_nb_generate_net.placeholders,state=cv_nb_generate_net.placeholders_state)
    
    
    # load weights
    session.run(tf.global_variables_initializer())

    for checkpoint in checkpoints:
        optimistic_restore(session,checkpoint,verbose=True)
        
    # read input data
    # with open(datafile,'rb') as f:
    #     sub_seq_py = pickle.load(f)

    key_image = Image.open(os.path.join(datafile, '1.png'))
    key_image = np.array(key_image)
    
    new_key_img = np.zeros((1,3, height, width))
    new_key_img[0, 0, :, :] = key_image[:height, :width] 

    new_key_img[0, 1, :, :] = key_image[:height, :width] 
    new_key_img[0, 2, :, :] = key_image[:height, :width]
    new_key_img = new_key_img/256
    # new_key_img = new_key_img.astype(np.uint8)

    axes = init_visualization('DeepTAM_Mapper')
    ######### depth_gt
    # depth_gt = sub_seq_py.get_depth(frame=0)


    ######### fixed band prediction with increasing number of frames
    depth_init = np.ones([1,1,height,width])

    for frame_id in range(1, 2):

        frame = frame_id
        # sub_sub_seq_py = SubSeqPy(sub_seq_py, start_frame=0, seq_len=frame+1)
 
        cv, depth_label_tensor = create_cv_conf_from_sequence_py(depth_init, datafile, session, cv_fb_generate_net, cv_fb_generate_outputs)
        np.save('cv.npy', cv)
        np.save('depth_label_tensor.npy', depth_label_tensor)
        print('max_cv', np.amax(cv))
        feed_dict = {
            fb_depth_net.placeholders['image_key']: new_key_img,
            fb_depth_net.placeholders_state['cv']:cv,
            fb_depth_net.placeholders_state['depth_label_tensor']:depth_label_tensor,
        }
        fb_out = session.run(fb_depth_outputs, feed_dict=feed_dict)

        depth_pr = fb_out['predict_depth']
        curr_image = Image.open(os.path.join(datafile, str(frame) + '.png'))
        curr_image = np.array(curr_image)
        curr_img_arr = np.zeros((1,3, height, width))
        curr_img_arr[0, 0, :, :] = curr_image[:height, :width] 

        curr_img_arr[0, 1, :, :] = curr_image[:height, :width] 
        curr_img_arr[0, 2, :, :] = curr_image[:height, :width]
        curr_img_arr = curr_img_arr/256
        # curr_img_arr = curr_img_arr.astype(np.uint8)

        # dm = compute_depth_metrics(1/depth_gt, 1/depth_pr)
        update_visualization(axes, 
                             new_key_img,
                             curr_img_arr,
                             depth_pr,
                             None,
                             ['',str(frame_id),str(frame_id),''])
    i = 0
    ######### narrow band prediction with increasing number of iterations 
    for frame_id in range(1, 2):
        frame = frame_id
        print('i', i)
        i += 1
        cv, depth_label_tensor = create_cv_conf_from_sequence_py(depth_pr, datafile, session, cv_nb_generate_net, cv_nb_generate_outputs)
        feed_dict = {
            nb_depth_net.placeholders['image_key']: new_key_img,
            nb_depth_net.placeholders_state['cv']:cv,
            nb_depth_net.placeholders_state['depth_label_tensor']:depth_label_tensor,
        }
        nb_out = session.run(nb_depth_outputs, feed_dict=feed_dict)
        depth_pr = nb_out['predict_depth']
        print('end of first nb', np.amax(depth_pr))
        feed_dict = {
            nb_refine_depth_net.placeholders['image_key']: new_key_img,
            nb_refine_depth_net.placeholders['depth_key']: depth_pr,
            nb_refine_depth_net.placeholders_state['cv']:cv,
        }
        nb_refine_out = session.run(nb_refine_depth_outputs, feed_dict=feed_dict)
        depth_pr = nb_refine_out['predict_depth'] 
        print('end of refine nb', np.amax(depth_pr))
        curr_image = Image.open(os.path.join(datafile, str(frame) + '.png'))

        curr_image = np.array(curr_image)
        curr_img_arr = np.zeros((1,3, height, width))
        curr_img_arr[0, 0, :, :] = curr_image[:height, :width] 

        curr_img_arr[0, 1, :, :] = curr_image[:height, :width] 
        curr_img_arr[0, 2, :, :] = curr_image[:height, :width]

        # curr_img_arr = curr_img_arr.astype(np.uint8)
        curr_image_arr = curr_image_arr/256
        # dm = compute_depth_metrics(1/depth_gt, 1/depth_pr)
        update_visualization(axes,
                             new_key_img,
                             curr_img_arr,
                             fb_out['predict_depth'], 
                             nb_refine_out['predict_depth'],
                             ['',str(frame_id),str(frame_id),str(0)])    
    
    plt.show()

    print('****************Saving image****************')
    plt.savefig('mapping_example.png')

    # time.sleep(1200)

    del session
    tf.reset_default_graph()



def main():

    
    examples_dir = os.path.dirname(__file__)
    mapping_module_path = os.path.join(examples_dir,'..','python/deeptam_mapper/models/networks.py')
    checkpoints = [os.path.join(examples_dir, '..', 'weights', 'deeptam_mapper_weights','snapshot-800000'),
                    ]

    sequence_len = 10
    nb_iterations_num = 5

    width = 320
    height = 240

    datafile = os.path.join(examples_dir,'..','data')
    mapping_with_pose(
                    datafile,
                    mapping_module_path,
                    checkpoints,
                    max_sequence_length=sequence_len,
                    nb_iterations_num=nb_iterations_num,
                    width=width,
                    height=height,
                    )
    
    
if __name__ == '__main__':

    main()