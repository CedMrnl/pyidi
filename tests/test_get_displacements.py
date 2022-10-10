import numpy as np
from numpy.testing import assert_array_equal
import sys, os
import matplotlib.pyplot as plt 
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi


def get_data(nb_img = 438):
    data = np.load("./data/video_force_1.3091147111765669_10s_stoch_False.npz")
    
    return data["video"][:nb_img], data["fps"]


def test_lk_scipy():
    nb_img = 438
    v, fps = get_data(nb_img)
    time = np.arange(nb_img, dtype=np.float64)/fps
    video = pyidi.pyIDI(v)

    video.set_method(method='lk_scipy')
    y = np.arange(20,700,40)
    points = np.column_stack((y, np.repeat(17,y.size)))
    video.set_points(points)
    
    video.method.configure(roi_size=(3,3))
    res_1 = video.get_displacements(video)
    res_1 -= np.mean(res_1,axis=1).reshape((res_1.shape[0],1,res_1.shape[2]))
    fig, ax = plt.subplots()
    ax.plot(time, res_1[...,1].T, color="white",lw=0.5, alpha=0.1)
    ax.set(xlim=(0,0.5), facecolor="black")
    plt.show()

def test_lk_scipy2():
    nb_img = 438
    v, fps = get_data(nb_img)
    time = np.arange(nb_img, dtype=np.float64)/fps
    video = pyidi.pyIDI(v)

    video.set_method(method='lk_scipy2')
    y = np.arange(20,700,40)
    points = np.column_stack((y, np.repeat(17,y.size)))
    video.set_points(points)
    
    video.method.configure(roi_size=(3,3))
    res_1 = video.get_displacements(video)
    res_1 -= np.mean(res_1,axis=1).reshape((res_1.shape[0],1,res_1.shape[2]))
    fig, ax = plt.subplots()
    ax.plot(time, res_1[...,1].T, color="white",lw=0.5, alpha=0.1)
    ax.set(xlim=(0,0.5), facecolor="black")
    plt.show()


def test_multiprocessing():
    data = np.load("./data/video_force_1.3091147111765669_10s_stoch_False.npz")

    # video = pyidi.pyIDI(data='./data/data_synthetic.cih')
    video = pyidi.pyIDI(data["video"])
    # video.set_method(method='lk', int_order=1, roi_size=(9, 9))
    

    # points = np.array([
    #     [ 31,  35],
    #     [ 31, 215],
    #     [ 31, 126],
    #     [ 95,  71],
    # ])
    # y = np.arange(620,625)
    # points = np.column_stack((y, np.repeat(17,y.size)))
    # video.set_points(points)
    # video.method.configure(pbar_type='tqdm', multi_type='multiprocessing')
    # res_1 = video.get_displacements(processes=2, resume_analysis=False, autosave=False)

    video.set_method(method='sof')
    points = np.array([
        [ 31,  35],
        [ 31, 215],
        [ 31, 126],
        [ 95,  71],
    ])
    y = np.arange(20,700,1)
    points = np.column_stack((y, np.repeat(17,y.size)))
    video.set_points(points)
    # video.method.configure(pbar_type='tqdm', multi_type='multiprocessing')
    video.method.configure(subset_size=1, reference_range=(0,1))
    res_1 = video.get_displacements(video)
    res_1 -= np.mean(res_1,axis=1).reshape((res_1.shape[0],1,res_1.shape[2]))
    fig, ax = plt.subplots()
    time = np.arange(data["video"].shape[0], dtype=np.float64)/data["fps"]
    ax.plot(time, res_1[...,1].T, color="white",lw=0.5, alpha=0.01)
    ax.set(xlim=(0,0.5), facecolor="black")
    plt.show()

    # video.method.configure(pbar_type='atpbar', multi_type='mantichora')
    # res_2 = video.get_displacements(processes=2, resume_analysis=False, autosave=False)

    # assert_array_equal(res_1, res_2)

if __name__ == "__main__":
    test_lk_scipy()
    test_multiprocessing()
    test_lk_scipy2()