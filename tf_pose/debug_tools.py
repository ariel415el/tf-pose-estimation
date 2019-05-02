import os
import cv2
import numpy as np
import sys
import common
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, parent_dir)

def draw_humans(npimg, humans, color=None, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]

    colors = np.random.rand(1,3)
    sks = np.array([(0,1),(1,2),(1,3), (2,4),(4,6), (3,5),(5,7), (1,8),(8,10),(10,12), (1,9),(9,11),(11,13)])
    centers = {}
    
    for i,kp in enumerate(humans):
        # draw point
        x = kp[0::3]
        y = kp[1::3]
        v = np.array(kp[2::3])
        if color is not None:
         person_color = color
        else:
            person_color = list(np.random.rand(1,3)[0]*255)
        # draw limbs
        for sk in sks:
            if np.all(v[sk] > 0):
                center_1 = (int(x[sk[0]] + 0.5), int(y[sk[0]] + 0.5))
                center_0 = (int(x[sk[1]] + 0.5), int(y[sk[1]] + 0.5))
                cv2.line(npimg, center_0, center_1, person_color, 2)
        for i in range(common.BC_parts.Background.value):
            if v[i] > 0:
                cv2.circle(npimg, (int(x[i] + 0.5), int(y[i] + 0.5)), 1, [0,0,0], thickness=1, lineType=4, shift=0)

    return npimg

def plot_from_csv(input_path, output_path, plotees):
    from itertools import cycle
    cycol = cycle('bgrcmk')

    import pandas as pd
    pd = pd.read_csv(input_path)
    step_nums = np.array(pd['Step_number'])
    for data_name in plotees:
        plt.plot(step_nums,  np.array(pd[data_name]), label=data_name, c=next(cycol))
    plt.xlabel('Step_number')
    plt.legend(loc='upper left')
    plt.savefig(output_path)
    plt.close()

def create_debug_collage(inp, upscaled_heatmap, upscaled_vectmaps, humans=None):
    global mplset
    mplset = True

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    if humans:
        debug_image = draw_humans(inp, humans, imgcopy=True)
        plt.imshow(debug_image)
    else:
        plt.imshow(inp)

    a = fig.add_subplot(2, 2, 2)
    a.set_title('Heatmap')
    plt.imshow(common.get_grey_img(inp, target_size=(upscaled_heatmap.shape[1], upscaled_heatmap.shape[0])), alpha=0.5, cmap='gray')
    tmp = np.amax(upscaled_heatmap, axis=2)*255
    plt.imshow(tmp, cmap='hot', alpha=0.5)
    plt.colorbar()

    tmp2 = upscaled_vectmaps.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)*255
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)*255

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    plt.imshow(common.get_grey_img(inp, target_size=(upscaled_vectmaps.shape[1], upscaled_vectmaps.shape[0])), alpha=0.5, cmap='gray')
    plt.imshow(tmp2_odd, cmap='hot', alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    plt.imshow(common.get_grey_img(inp, target_size=(upscaled_vectmaps.shape[1], upscaled_vectmaps.shape[0])), alpha=0.5, cmap='gray')
    plt.imshow(tmp2_even, cmap='hot', alpha=0.5)
    plt.colorbar()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clear()
    plt.close()
    return data
