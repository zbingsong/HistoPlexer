import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from shapely.affinity import scale
from shapely.ops import unary_union
from PIL import ImageDraw
import os
import pandas as pd
import json
from types import SimpleNamespace
import tifffile
from src.models.generator import unet_translator

class HistoplexerInferenceWSI:
    def __init__(self, checkpoint_path, seg_level, chunk_size, batch_size, protein_subset, loader_kwargs, device, normalizer):
        self.checkpoint_path = checkpoint_path
        self.seg_level = seg_level
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.protein_subset = protein_subset
        self.n_proteins = len(protein_subset)
        self.loader_kwargs = loader_kwargs
        self.device = device
        self.normalizer = normalizer
        
        # loading model and checkpoint        
        config_path = os.path.dirname(self.checkpoint_path) + '/config.json'        
        with open(config_path, "r") as f:
            self.config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
                
        self.model = unet_translator(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            use_high_res=self.config.use_high_res,
            use_multiscale=self.config.use_multiscale,
            ngf=self.config.ngf,
            depth=self.config.depth,
            encoder_padding=self.config.encoder_padding,
            decoder_padding=self.config.decoder_padding, 
            device="cpu", 
            extra_feature_size=self.config.fm_feature_size
        )
        print("Model created!")

        # load model weights all in cpu
        self.checkpoint_name = os.path.basename(self.checkpoint_path).split('.')[0]
        print(f"Checkpoint name: {self.checkpoint_name}")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['trans_ema_state_dict']) # trans_state_dict
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded!")
            

    def segment_tissue(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mthresh = 7
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
        _, img_prepped = cv2.threshold(
            img_med, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

        close = 4
        kernel = np.ones((close, close), np.uint8)
        img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

        # Find and filter contours
        contours, hierarchy = cv2.findContours(
            img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

    def detect_foreground(self, contours, hierarchy):
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # find foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

        all_holes = []
        for cont_idx in hierarchy_1:
            all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

        hole_contours = []
        for hole_ids in all_holes:
            holes = [contours[idx] for idx in hole_ids]
            hole_contours.append(holes)

        return foreground_contours, hole_contours

    def construct_tissue_polygon(self, foreground_contours, hole_contours, min_area):
        polys = []
        for foreground, holes in zip(foreground_contours, hole_contours):
            # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
            if len(foreground) < 3:
                continue

            # remove redundant dimensions from the contour and convert to Shapely Polygon
            poly = Polygon(np.squeeze(foreground))

            # discard all polygons that are considered too small
            if poly.area < min_area:
                continue

            if not poly.is_valid:
                # This is likely becausee the polygon is self-touching or self-crossing.
                # Try and 'correct' the polygon using the zero-length buffer() trick.
                # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
                poly = poly.buffer(0)

            # Punch the holes in the polygon
            for hole_contour in holes:
                if len(hole_contour) < 3:
                    continue

                hole = Polygon(np.squeeze(hole_contour))

                if not hole.is_valid:
                    continue

                # ignore all very small holes
                if hole.area < min_area:
                    continue

                poly = poly.difference(hole)

            polys.append(poly)

        if len(polys) == 0:
            raise Exception("Raw tissue mask consists of 0 polygons")

        # If we have multiple polygons, we merge any overlap between them using unary_union().
        # This will result in a Polygon or MultiPolygon with most tissue masks.
        return unary_union(polys)

    def make_tile_QC_fig(self, tile_sets, slide, level, line_width_pix):
        # Render the tiles on an image derived from the specified zoom level
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        downsample = 1 / slide.level_downsamples[level]

        draw = ImageDraw.Draw(img, 'RGBA')
        colors = ['red', 'green']
        assert len(tile_sets) <= len(colors), 'define more colors'
        for tiles, color in zip(tile_sets, colors):
            for tile in tiles:
                bbox = tuple(np.array(tile.bounds) * downsample)
                # bbox = tuple(np.array(tile.bounds))
                draw.rectangle(bbox, outline=color, width=line_width_pix)

        img = img.convert('RGB')
        return img

    def create_tissue_mask(self, wsi, min_rel_surface_area=500):
        # Determine the best level to determine the segmentation on
        level_dims = wsi.level_dimensions[self.seg_level]
        print(f"Segmentation level: {self.seg_level}, dimensions: {level_dims}")

        img = np.array(wsi.read_region((0, 0), self.seg_level, level_dims))
        contours, hierarchy = self.segment_tissue(img)
        if hierarchy is None or len(contours) == 0:
            raise ValueError(
                f"No contours detected at seg_level={self.seg_level}. "
                f"Try a finer segmentation level or a different min_rel_surface_area."
            )
        foreground_contours, hole_contours = self.detect_foreground(contours, hierarchy)

        # Get the total surface area of the slide level that was used
        level_area = level_dims[0] * level_dims[1]

        # Minimum surface area of tissue polygons (in pixels)
        min_area = level_area / min_rel_surface_area

        tissue_mask = self.construct_tissue_polygon(
            foreground_contours, hole_contours, min_area)

        # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
        scale_factor = wsi.level_downsamples[self.seg_level]
        tissue_mask_scaled = scale(
            tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0))

        return tissue_mask_scaled

    def create_tiles_in_mask(self, wsi, tissue_mask_scaled, tile_size_pix, stride, padding=0):
        # Generate tiles covering the entire mask
        minx, miny, _, _ = tissue_mask_scaled.bounds
        print('create_tiles_in_mask', tissue_mask_scaled.bounds, tile_size_pix, stride, padding)

        # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
        maxx, maxy = wsi.level_dimensions[0]
        cols = range(int(minx), int(maxx-tile_size_pix), stride)
        rows = range(int(miny), int(maxy-tile_size_pix), stride)
        rects = []
        for x in cols:
            for y in rows:
                # (minx, miny, maxx, maxy)
                rect = box(
                    x - padding,
                    y - padding,
                    x + tile_size_pix + padding,
                    y + tile_size_pix + padding,
                )

                # Retain only the tiles that partially overlap with the tissue mask.
                if tissue_mask_scaled.intersects(rect):
                    rects.append(rect)
                    # print(f'input of box:', x-padding, y-padding, x+tile_size_pix+padding, y+tile_size_pix+padding)
                    # print(f'shape of rect: {rect.bounds}, area of rect: {rect.area}')

        return rects

    def infer_batch(self, batch_imgs, model):
        batch_imgs = batch_imgs.to(self.device, non_blocking=True)
        with torch.no_grad():
            pred_dict = model(batch_imgs)
            # Restructure the tensor: move the 'values' to the last dimension.
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()]
                 for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            pred_output = torch.cat(list(pred_dict.values()), -1)
        return pred_output.cpu().numpy()

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_chunks(self, wsi):
        last_exc = None
        min_rel_surface_area_candidates = [500, 1000, 2000]
        print('chunk_size: ', self.chunk_size)

        for min_rel_surface_area in min_rel_surface_area_candidates:
            try:
                tissue_mask = self.create_tissue_mask(
                    wsi, min_rel_surface_area=min_rel_surface_area
                )
                # print(f'chunk size after tissue mask creation: {self.chunk_size}')
                chunks = self.create_tiles_in_mask(
                    wsi,
                    tissue_mask,
                    tile_size_pix=self.chunk_size,
                    stride=self.chunk_size,
                )
                if len(chunks) == 0:
                    raise ValueError(
                        f"No tiles found at seg_level={self.seg_level}, "
                        f"min_rel_surface_area={min_rel_surface_area}."
                    )
                print(
                    f"Tissue mask created with seg_level={self.seg_level}, "
                    f"min_rel_surface_area={min_rel_surface_area}, chunks={len(chunks)}",
                    f"first chunk bounds: {chunks[0].bounds if len(chunks) > 0 else 'N/A'}"
                )
                return tissue_mask, chunks
            except Exception as exc:
                last_exc = exc
                print(
                    f"Tissue mask attempt failed at seg_level={self.seg_level}, "
                    f"min_rel_surface_area={min_rel_surface_area}: {exc}"
                )

        raise RuntimeError(
            "Unable to create a valid tissue mask after trying multiple min_rel_surface_area values."
        ) from last_exc

    def get_qc_img(self, wsi, chunks):
        qc_img = self.make_tile_QC_fig([chunks], wsi, self.seg_level, 1)
        qc_img_target_width = 1920
        qc_img = qc_img.resize((qc_img_target_width, int(
            qc_img.height / (qc_img.width / qc_img_target_width))))
        return qc_img

    def save_qc_and_coords(self, sample, chunks, qc_img, save_path_imgs):
        path_coords = save_path_imgs.joinpath('coords')
        self.create_dir(path_coords)
        pd.DataFrame([rect.bounds for rect in chunks], columns=['xmin', 'ymin', 'xmax', 'ymax']).to_csv(
            path_coords.joinpath(sample + 'coords.csv'), index=False)

        path_qc = save_path_imgs.joinpath('QC')
        self.create_dir(path_qc)
        qc_img.save(path_qc.joinpath(sample + '_tile_QC.png'))

    def _get_nearest_level_for_downsample(self, wsi, target_downsample):
        """Return the pyramid level with downsample closest to target_downsample."""
        level_downsamples = np.array(wsi.level_downsamples, dtype=np.float64)
        return int(np.argmin(np.abs(level_downsamples - target_downsample)))

    def _resize_prediction_hwc(self, pred_hwc, target_width, target_height):
        """Resize an HWC prediction map to a target slide level size."""
        src_h, src_w = pred_hwc.shape[:2]
        target_width = int(target_width)
        target_height = int(target_height)

        if src_w == target_width and src_h == target_height:
            return pred_hwc.astype(np.float32, copy=False)

        if target_width < src_w or target_height < src_h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        resized = cv2.resize(
            pred_hwc,
            (target_width, target_height),
            interpolation=interpolation,
        )

        if resized.ndim == 2:
            resized = resized[..., np.newaxis]

        return resized.astype(np.float32, copy=False)

    def get_wsi_inference(self, sample, wsi, save_path_imgs):
        # imc multiplex to be generated at 3 downsamples
        print('wsi.level_downsamples:', wsi.level_downsamples)
        print('wsi.level_dimensions:', wsi.level_dimensions)
        # self.seg_level = wsi.get_best_level_for_downsample(64)

        # if 'TCGA' in sample:
        #     inference_downsample_target = 4.01
        # else:
        #     inference_downsample_target = 4.0

        # Using nearest level avoids falling back to level 0 when level-1 downsample is ~4.00x.
        # level = self._get_nearest_level_for_downsample(wsi, inference_downsample_target)
        print(f"selected segmentation level: {self.seg_level}, downsample: {wsi.level_downsamples[self.seg_level]}, segmentation level dims: {wsi.level_dimensions[self.seg_level]}")

        top_level_dims = wsi.level_dimensions[0]

        # There is a 4x downsample due to U-Net architecture, which has 7 levels of downsampling and 5 levels of upsampling
        # so top-level predictions are at 4x downsample, and we will resize to other levels from there.
        imc_pred_4x_downsample = np.zeros((top_level_dims[1] // 4, top_level_dims[0] // 4, self.n_proteins), dtype=np.float32)
        imc_pred_16x_downsample = np.zeros((top_level_dims[1] // 16 ,top_level_dims[0] // 16, self.n_proteins), dtype=np.float32)
        imc_pred_64x_downsample = np.zeros((top_level_dims[1] // 64 ,top_level_dims[0] // 64, self.n_proteins), dtype=np.float32)

        _, chunks = self.get_chunks(wsi) # get chunks
        qc_img = self.get_qc_img(wsi, chunks) # get wsi with tiling for qc 
        
        # saving coordinates of tiles
        self.save_qc_and_coords(sample, chunks, qc_img, save_path_imgs)
        print(len(chunks))

        loader_kwargs = dict(self.loader_kwargs) if self.loader_kwargs is not None else {}

        loader = DataLoader(
            dataset=BagOfTiles(wsi, chunks, self.normalizer),
            batch_size=self.batch_size,
            **loader_kwargs,
        )
        print('dataloader size:', len(loader))

        for batch_id, (batch, coord) in enumerate(loader):
            # predict for batch 
            with torch.no_grad():
                # batch = (batch/255.0).to(dev0)
                batch = batch.to(self.device)
                print(batch_id, batch.shape)
                imc_batch = self.model(batch)
                for i, imc in enumerate(imc_batch):
                    print(i, imc.shape)
                # shapes of imc_batch list: (batch_size, 11, 16, 16), (batch_size, 11, 64, 64), (batch_size, 11, 256, 256)
                # Input is (batch_size, 3, 1024, 1024), so 4x downsampling

            coord = (coord.detach().cpu().numpy())#.astype(int)
            for i, c in enumerate(coord):
                if any(x<0 for x in c) == True:
                    pass
                else:
                    # print(f'coord of tile {i}: ', c, 'imc_batch[2][i].shape: ', imc_batch[2][i].shape)
                    c = c // 4 # to match the 4x downsampled prediction map
                    imc_pred_4x_downsample[c[1]:c[3], c[0]:c[2], :] = (torch.permute(imc_batch[2][i], (1, 2, 0))).detach().cpu().numpy()
                    c = c // 4 # to match the 16x downsampled prediction map
                    imc_pred_16x_downsample[c[1]:c[3], c[0]:c[2], :] = (torch.permute(imc_batch[1][i], (1, 2, 0))).detach().cpu().numpy()
                    c = c // 4 # to match the 64x downsampled prediction map
                    imc_pred_64x_downsample[c[1]:c[3], c[0]:c[2], :] = (torch.permute(imc_batch[0][i], (1, 2, 0))).detach().cpu().numpy()

        
        print('top level (4x downsample):', imc_pred_4x_downsample.shape)
        print('16x downsample:', imc_pred_16x_downsample.shape)
        print('64x downsample:', imc_pred_64x_downsample.shape)

        # Save predictions for every native slide pyramid level.
        # We stitch at the selected inference level and then resize to each level dimension.
        # path_all_levels = save_path_imgs.joinpath('all_levels')
        # self.create_dir(path_all_levels)

        tiff_output_path = save_path_imgs.joinpath(sample + '.ome.tif')
        n_subifds = max(0, len(wsi.level_dimensions) - 1)

        with tifffile.TiffWriter(tiff_output_path, bigtiff=True, ome=True) as tif:
            # for level_idx in range(level, len(wsi.level_dimensions)):
            #     if level_idx == level:
            #         cyx_level_i = hwc_to_cyx(imc_pred_4x_downsample)
            #         tif.write(
            #             cyx_level_i,
            #             metadata={
            #                 "axes": "CYX",
            #                 "Channel": {"Name": self.protein_subset},
            #             },
            #             subifds=n_subifds,
            #             compression="zlib",
            #         )
            #     else:
            #         downsample_factor = wsi.level_downsamples[level_idx] / wsi.level_downsamples[level]
            #         if abs(downsample_factor - 4.0) < 0.05:
            #             imc_pred_level_i = imc_pred_16x_downsample
            #         elif abs(downsample_factor - 16.0) < 0.05:
            #             imc_pred_level_i = imc_pred_64x_downsample
            #         else:
            #             level_dims_i = wsi.level_dimensions[level_idx]
            #             imc_pred_level_i = self._resize_prediction_hwc(
            #                 imc_pred_4x_downsample,
            #                 target_width=level_dims_i[0],
            #                 target_height=level_dims_i[1],
            #             )
            #         cyx_level_i = hwc_to_cyx(imc_pred_level_i)
            #         # np.save(
            #         #     path_all_levels.joinpath(sample + f'.level_{level_idx}.npy'),
            #         #     pred_level_i,
            #         # )
            #         tif.write(
            #             cyx_level_i,
            #             subfiletype=1,
            #             metadata={"axes": "CYX"},
            #             compression="zlib",
            #         )
            # write the 4x downsampled prediction map as the main image
            cyx_level_0 = hwc_to_cyx(imc_pred_4x_downsample)
            tif.write(
                cyx_level_0,
                metadata={
                    "axes": "CYX",
                    "Channel": {"Name": self.protein_subset},
                },
                subifds=n_subifds,
                compression="zlib",
            )
            cyx_level_1 = hwc_to_cyx(imc_pred_16x_downsample)
            tif.write(
                cyx_level_1,
                subfiletype=1,
                metadata={"axes": "CYX"},
                compression="zlib",
            )
            cyx_level_2 = hwc_to_cyx(imc_pred_64x_downsample)
            tif.write(
                cyx_level_2,
                subfiletype=1,
                metadata={"axes": "CYX"},
                compression="zlib",
            )

        print(f"Saved: {tiff_output_path}")


def hwc_to_cyx(arr):
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {arr.shape}")
    return np.moveaxis(arr, -1, 0).astype(np.float32, copy=False)


class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles, normalizer=None):
        self.wsi = wsi
        self.tiles = tiles

        self.stain_normalizer = normalizer
        self.to_tensor = transforms.ToTensor()
        self.to_byte = transforms.Lambda(lambda x: x*255)
        self.to_unit = transforms.Lambda(lambda x: x / 255.0)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = self.crop_rect_from_slide(self.wsi, tile)

        # Convert from RGBA to RGB
        img = img.convert('RGB')

        # Ensure we have a square tile in our hands.
        width, height = img.size
        assert width == height, 'input image is not a square'

        # Turn the PIL image into a (C x H x W) torch.FloatTensor (32 bit by default)
        img = self.to_tensor(img)

        if self.stain_normalizer:
            img = self.to_byte(img)
            img, _, _ = self.stain_normalizer.normalize(I=img, stains=False) # return shape: [H, W, C], range: [0, 255]
            img = img.permute(2, 0, 1) # [H, W, C] --> [C, H, W]
            img = self.to_unit(img)

        # img = img * 255 # to range 0 to 255 
        coords = np.array(tile.bounds).astype(np.int32)
        return img, coords
    
    @staticmethod
    def crop_rect_from_slide(slide, rect):
        minx, miny, maxx, maxy = rect.bounds
        # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
        # but in the slide it is y = 0. Hence: miny instead of maxy.
        top_left_coords = (int(minx), int(miny))
        return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))
