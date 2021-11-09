import os
import time
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [20,5]

class_name = [
    'BICYCLE',
    'BUS',
    'CAR',
    'CYCLIST',
    'MOTORCYCLE',
    'PEDESTRIAN',
    'TRUCK'
]
    
class_marker_shape = {
    'BICYCLE': '8',
    'BUS': 'p',
    'CAR': 'X',
    'CYCLIST': 'o',
    'MOTORCYCLE': 'D',
    'PEDESTRIAN': '*',
    'TRUCK': 'v',
}

file_path = os.path.realpath(__file__)
file_dir = os.path.dirname(file_path)
cyclist_img = mpimg.imread(os.path.join(file_dir,'cyclist-top.png'))

def plot_birds_eye(data, save_every_plot=False):
    '''
    Matplotlib plot of approaching objects from the outputs of Intercept
    
    args:
        data: a dictionary containing all detection info
            example:
            {11.0: {'raw_x': 19.8,
                    'raw_y':0.26,
                    'class_index':3.0,
                    'TTE': array([15.6]),
                    'EP': array([-4.1]),
                    'filter_pred_x':array([22.1]),
                    'filter_pred_y':...
                    'filter_pred_dxdt':...
                    'filter_pred_dydt':...
                    'dangerous':False,
                    },
             ...
             }
             
    returns matplotlib image
    
    '''
    if len(data) < 1:
        return

    imagebox = OffsetImage(cyclist_img, zoom=0.08)

    ab = AnnotationBbox(imagebox, (0.0, 0.0), frameon=False)
    
    fig, ax = plt.subplots()
    plt.axis([-100,100,-10,10])
    plt.xlabel('FORWARD(+) / REAR(-) distance')
    plt.ylabel('LEFT(+) / RIGHT(-) distance')
    ax.set_facecolor('silver')

    ax.add_artist(ab)

    # vertical line representing overtake line
    plt.axline((0,0),(0,1), alpha=0.2, color='red', linestyle='--', linewidth=5)

    # horizontal line representing road
    plt.axline((0,0),(1,0), alpha=1.0, color='white', linestyle='-', linewidth=130)

    for track_id in data:
        point = data[track_id]
        filter_x = point['filter_pred_x']
        filter_y = point['filter_pred_y']
        filter_dxdt = point['filter_pred_dxdt']
        filter_dydt = point['filter_pred_dydt']
        TTE = float(point['TTE']) if point['TTE'] is not None else 999
        EP = float(point['EP']) if point['EP'] is not None else 999

        # filter predictions arrow
        plt.arrow(
            filter_x, 
            filter_y, 
            3*filter_dxdt, 
            3*filter_dydt, 
            width=1, 
            head_width=2, 
            head_length=1.5, 
            alpha=0.6, 
            fill=True,
            color = 'orangered' if point['dangerous'] else 'skyblue',
            zorder=999)

        # filter prediction arrow
        plt.plot(
            filter_x, 
            filter_y, 
            color = 'orangered' if point['dangerous'] else 'skyblue',
            marker=class_marker_shape[class_name[int(point['class_index'])]],
            markersize=20.0,
            alpha=0.6,
            zorder=999
        )

        # 'CYCLIST 11' text
        plt.annotate(
            f"{class_name[int(point['class_index'])]} {int(track_id)}",
            (filter_x-3, filter_y+2),
            fontweight=600,
            fontsize=15.0,
            alpha=0.8
            
        )

        # TTE text
        plt.annotate(
            f"{round(TTE,1)}s away",
            (filter_x-2.5, filter_y-3),
            fontweight=400,
            fontsize=15.0,
            alpha=0.8,
        )

        # raw current point
        plt.plot(
            point['raw_x'],
            point['raw_y'],
            color = 'maroon' if point['dangerous'] else 'steelblue',
            marker=class_marker_shape[class_name[int(point['class_index'])]],
            markersize=20.0,
            alpha=0.9,
            zorder=999
        )
        
    if save_every_plot:
        fig.savefig(os.path.join(file_dir, "saved_figs", f"{time.time()}.png"))
    
    # image from plot for viewing with opencv
    #fig.tight_layout(pad=0)
    #ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    plt.clf()
    plt.cla()
    return image_from_plot
