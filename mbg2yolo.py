import argparse, os, cv2
import pandas as pd
from lxml import etree


if __name__ == '__main__':
    
    ###### step 1: build dataframe with all annotated samples (frames)
    
    parser = argparse.ArgumentParser(prog='mbg2yolo.py')
    parser.add_argument('--path', default='/dataset', help='dataset source path')
    opt = parser.parse_args()
    
    datapath = opt.path
    
    dest_folder = 'data'
    if not os.path.isdir(dest_folder): os.mkdir(dest_folder)
    savepath = os.path.join(dest_folder, 'annotations.csv')

    print(f'------- Read xml annotations from *all files* at {datapath}-------')
    print('\t - Make sure you are referencing a folder containing test samples only!')
    
    if os.path.isfile(savepath):
        print(f'\t - Loading cached {savepath}')
        df_labels = pd.read_csv(savepath, index_col=0)
    else:
        print(f'\t - Creating {savepath} with annotations from all files at {datapath}')
        df_labels = {'videopath': [], 'num_frames': [], 'w': [], 'h': [], ## video
                     'label': [], 'frame': [], 'x0': [], 'y0': [], 'x1': [], 'y1': []  ## frames
                    }

        sample_indices = [0] 
        for s, sample in enumerate(sorted(os.listdir(datapath))):

            if not os.path.isdir(os.path.join(datapath, sample)): continue

            ann_path = os.path.join(datapath, sample, 'ann/')
            if not os.path.isdir(ann_path): 
                raise ValueError(f'Annotations not found at {ann_path}')

            ann_path += os.listdir(ann_path)[0]
            
            vid_path = os.path.join(datapath, sample, 'avi/')
            try:
                vid_path = os.path.join(vid_path, os.listdir(vid_path)[0])
            except:
                raise ValueError(f'Video not found at {vid_path}')

            root = etree.parse(ann_path).getroot()
            for t in range(2, len(root)):
                track = root[t]

                for b in range(len(track)):
                    box = track[b]

                    df_labels['videopath']  .append(vid_path)
                    df_labels['num_frames'] .append(int(root[1][0][2].text))
                    df_labels['w']          .append(int(root[1][0][15][0].text))
                    df_labels['h']          .append(int(root[1][0][15][1].text))

                    df_labels['label']    .append(track.get('label')) 
                    df_labels['frame']    .append(box.get('frame'))
                    df_labels['x0']       .append(int(box.get('xtl').split('.')[0]))
                    df_labels['y0']       .append(int(box.get('ytl').split('.')[0]))
                    df_labels['x1']       .append(int(box.get('xbr').split('.')[0])) 
                    df_labels['y1']       .append(int(box.get('ybr').split('.')[0]))
            sample_indices.append(len(df_labels['videopath']))

        df_labels = pd.DataFrame(df_labels)
        df_labels.to_csv(savepath)
    print('Done')
    
    ####### step 2: extract annotated frames and generate yolo labels
    print('------- Extract frames and generate yolo labels (it may take a while) -------')
    
    output_folder = f'{dest_folder}/yolo_dataset'
    images_folder = f'{output_folder}/images'
    annot_folder  = f'{output_folder}/labels'
    samples_path  = f'{dest_folder}/samples.txt'

    if not os.path.isdir(output_folder): os.mkdir(output_folder)
    if not os.path.isdir(images_folder): os.mkdir(images_folder)
    if not os.path.isdir(annot_folder):  os.mkdir(annot_folder)

    labels = {'bucket': 0, 'watertank': 1, 'bottle': 2, 'pool': 3, 'tire': 4, 'puddle': 5}
    filenames = ''

    for k, videopath in enumerate(df_labels['videopath'].unique()):
        print(f'\t - {videopath}')

        video = cv2.VideoCapture(videopath)
        df_sample = df_labels[df_labels['videopath'] == videopath]
        frames = sorted(df_sample.frame.unique())

        basename = os.path.basename(videopath)
        for frame_id in frames: 
            out_filename = f'{basename.split(".")[0]}_{frame_id}'
            filenames += f'{images_folder}/{out_filename}.jpg\n'
            
            if not os.path.isfile(f'{images_folder}/{out_filename}.jpg'):
                video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id) )
                ret, frame = video.read()
                cv2.imwrite(f'{images_folder}/{out_filename}.jpg',frame)

            if not os.path.isfile(f'{annot_folder}/{out_filename}.txt'):
                d = df_sample[df_sample.frame == frame_id]
                annot = ''
                for obj in range(len(d)):
                    x0, y0 = d.iloc[obj]['x0'].item(), d.iloc[obj]['y0'].item()
                    x1, y1 = d.iloc[obj]['x1'].item(), d.iloc[obj]['y1'].item()
                    res = (d.iloc[obj]['w'], d.iloc[obj]['h'])
                    label = d.iloc[obj]['label']

                    w, h = (x1-x0), (y1-y0)
                    xc, yc = (x0 + (w/2)), (y0 + (h/2))
                    xc, yc, w, h = xc/res[0], yc/res[1], w/res[0], h/res[1]

                    annot += f'{labels[label]} {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}\n'

                
                with open(f'{annot_folder}/{out_filename}.txt', 'w') as fp:
                    fp.write(annot)
    
    
    print(f'------- Step 3: writing {samples_path} with all samples for testing -------')
    with open(samples_path, 'w') as fp:
        fp.write(filenames)
    print("Done")

