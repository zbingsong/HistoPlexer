import os 
import openslide 
import glob
import argparse
import time 
import json
from pathlib import Path
from PIL import ImageDraw, Image
from torchvision import transforms
import torchstain
from types import SimpleNamespace
import json

from src.inference.histoplexer_inference_wsi import HistoplexerInferenceWSI

parser = argparse.ArgumentParser(description='HistoPlexer prediction on whole slide images')
parser.add_argument("--checkpoint_path", type=str, required=False, default=None, help="Path to checkpoint file")
parser.add_argument('--wsi_paths', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/HE_new_wsi', help='path where wsi h&e images reside')
parser.add_argument('--sample', type=str, required=False, default=None, help='sample name for which we need to run wsi prediction')
parser.add_argument("--device", type=str, required=False, default='cuda:0', help="device to use")
parser.add_argument('--data_set', type=str, required=False, default="test", help='Which data_set to use {test, external_test, HE_ultivue}')
parser.add_argument('--ref_img_path', type=str, required=False, default=None, help='Path to reference image for stain normalization')
parser.add_argument('--chunk_size', type=int, required=False, default=1024, help='the tile size used for inference')
parser.add_argument('--batch_size', type=int, required=False, default=128, help='the batch size used for inference')
parser.add_argument('--save_path', type=str, required=False, default=None, help='the path used for saving')

# ----- paths etc -----
args = parser.parse_args()
data_set = args.data_set
wsi_paths = args.wsi_paths

if args.save_path is None:
    save_path = Path(os.path.dirname(args.checkpoint_path))  
else:
    os.makedirs(args.save_path, exist_ok=True)
    save_path = Path(args.save_path)
print('save_path: ', save_path) 

# ----- getting config for experiment -----
config_path = os.path.dirname(args.checkpoint_path) + '/config.json'        
with open(config_path, "r") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
protein_subset = config.markers
print('protein_subset: ', len(protein_subset), protein_subset)

# ----- defining specifics for inference -----
chunk_size = args.chunk_size 
chunk_padding = 0
batch_size = args.batch_size
loader_kwargs = {'num_workers': 2, 'pin_memory': False}

# ----- prepare reference img for stain normalization -----
normalizer = None
if args.ref_img_path:
    print("Initilize MacenkoNormalizer...")
    ref_img = Image.open(args.ref_img_path)
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(tsfm(ref_img))
    
# ----- Initialize HistoplexerInferenceWSI -----
histoplexer_wsi = HistoplexerInferenceWSI(
    checkpoint_path=args.checkpoint_path,
    seg_level=1,  # Default segmentation level
    chunk_size=chunk_size,
    batch_size=batch_size,
    protein_subset=protein_subset,
    loader_kwargs=loader_kwargs,
    device=args.device,
    normalizer=normalizer
)

# ----- get sample_roi names for a cv split from experiment config -----
if args.sample is None: 
    samples = glob.glob(wsi_paths + '/*.svs')
    print('samples: ', samples)
    
    for sample in samples: 
        sample = sample.split('/')[-1].split('.svs')[0]
        print('sample: ', sample)
        save_path_imgs = save_path.joinpath(sample + "_wsis")
        try: 
            wsi_path = glob.glob(wsi_paths + '/' + sample + '.svs')[0]
            wsi = openslide.open_slide(wsi_path)
        except:
            print('no WSI image found for sample ', sample)
            continue
        
        print('save file exists: ', os.path.isfile(os.path.join(save_path_imgs, sample + '.ome.tif')))
        if not os.path.isfile(os.path.join(save_path_imgs, sample + '.ome.tif')): 
            start_time = time.time()
            histoplexer_wsi.get_wsi_inference(sample, wsi, save_path_imgs)

            # timing
            end_time = time.time()
            print('time for wsi: ', end_time-start_time)
            hours, rem = divmod(end_time-start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        else: 
            print('Inference already done for: ', sample)

else: 
    # predict for only chosen sample
    sample = args.sample.split('.svs')[0]
    save_path_imgs = save_path.joinpath(sample + "_wsis")
    
    # check if inference is already done for this sample
    if not os.path.isfile(os.path.join(save_path_imgs, sample + '.ome.tif')):
        if (len(glob.glob(wsi_paths + '/' + sample + '.svs')) != 0):
            wsi_path = glob.glob(wsi_paths + '/' + sample + '.svs')[0]
            wsi = openslide.open_slide(wsi_path)

            start_time = time.time()
            histoplexer_wsi.get_wsi_inference(sample, wsi, save_path_imgs)

            # timing
            end_time = time.time()
            print('time for wsi: ', end_time-start_time)
            hours, rem = divmod(end_time-start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        else: 
            print('no WSI image found for sample ', args.sample)
    else: 
        print('Inference already done for: ', sample)
