import os, sys, math
import torch, cv2
import numpy as np
from torchvision import transforms

rootpath = '.'
sys.path.append(f'{rootpath}/yolo/')
from utils.datasets import letterbox
from utils.general import scale_coords, non_max_suppression_kpt, bbox_iou

def attempt_track(objects, newobj, thrs_iou = 0.5):
    iou_lst = []
    for idx, lst in objects.items():
        lastseen = lst[-1]
        iou = bbox_iou(torch.tensor(lastseen), torch.tensor(newobj), CIoU=True)
        iou_lst.append( (iou, idx) )

    max_iou = max(iou_lst, key=lambda x: x[0])
    if max_iou[0] > thrs_iou:
        return max_iou[1]
    else:
        return -1
            

def track_objects(rootpath):

    tracking = {}
    last_id  = 0 
    frames = sorted(os.listdir(rootpath))
    for f, frame in enumerate(frames):
        if f%100 == 0:
            print(f'\r{f}/{len(frames)}', end='', flush=True)
        
        filepath = os.path.join(rootpath, frame)
        fp = open(filepath, 'r')
        labels = fp.read().split('\n')[:-1]
        fp.close()

        new_labels = ''
        for lab in labels:
            new_labels += lab[0] + ' ' ##
            
            lab = np.array(lab.split(' ')).astype(float)
            clas = lab[0]
            if clas in tracking:
                idx = attempt_track(tracking[clas], lab[1:5])
                if idx != -1:
                    tracking[clas][idx].append(lab[1:5])
                else:
                    tracking[clas][last_id] = [lab[1:5]]
                    idx = last_id
                    last_id += 1
            else:
                tracking[clas] = {}
                tracking[clas][last_id] = [lab[1:5]]
                idx = last_id
                last_id += 1
                
            new_labels += str(idx) + ' ' ##
            new_labels += ' '.join([str(l) for l in lab[1:]]) + '\n' ##
        
        fp = open(filepath, 'w')
        fp.write(new_labels)
        fp.close()
        
    return tracking


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='mbg2yolo.py')
    parser.add_argument('--datapath', default='/dataset', help='dataset source path')
    parser.add_argument('--modelpath', default='weights/yolov7_mbg_adam_960_alltrain', help='path to trained weights')
    parser.add_argument('--destpath', default='.', help='path to store outputs from this script')
    opt = parser.parse_args()
    
    datapath  = opt.datapath
    modelpath = opt.modelpath 
    destpath  = opt.destpath
    
    tempfolder = f'{destpath}/yolo_labels/'
    if not os.path.isdir(tempfolder): os.mkdir(tempfolder)
        
    print(f'------ Loading {modelpath}')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(modelpath, map_location=device)
    model = weigths['model']

    _ = model.float().eval()
    if torch.cuda.is_available():
        model.to(device)
        
    print(f'------ Processing videos at {datapath}')
    for s, sample in enumerate(sorted(os.listdir(datapath))):
        
        if not sample.split('.')[-1] == 'avi': continue
        print(sample)

        outpath = os.path.join(tempfolder, sample.split('.')[0])
        if not os.path.isdir(outpath): os.mkdir(outpath)

        vid_path = os.path.join(datapath, sample)
        video = cv2.VideoCapture(vid_path)
        if not video.isOpened(): 
            print(f"Error opening {vid_path}")
            continue

        count = 0
        while(video.isOpened()):

            ret, frame = video.read()
            if not ret: break

            image, ratio, pad = letterbox(frame, 960, auto=False)
            image = image[:, :, ::-1]  
            image = np.ascontiguousarray(image)

            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            if torch.cuda.is_available():
                image = image.to(device)

            output, _ = model(image)
            output = non_max_suppression_kpt(output, conf_thres=0.5, iou_thres=0.5, 
                                             nc=model.yaml['nc'], kpt_label=False)

            output = output[0]
            if len(output) > 0:

                output = scale_coords(image.shape, output, frame.shape, [ratio,pad])
                outfile = f"{sample.split('.')[0]}_{count:0>5}.txt"
                outfile = os.path.join(outpath, outfile)

                outstr = ''
                for out in output:
                    outstr += str(int(out[-1].item())) + ' ' 
                    outstr += ' '.join([f'{str(o.item()):.9}' for o in out[:-1]]) + '\n'

                fp = open(outfile, 'w')
                fp.write(outstr)
                fp.close()

            count += 1
        video.release()
    
    print('------ Tracking instances', end='')
    for sample in range(1,len(13)):
        print(f'\nicip_mbg_test_{sample:0>2}')
        tracking = track_objects(f'{tempfolder}icip_mbg_test_{sample:0>2}')

