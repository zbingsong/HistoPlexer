# How to

## Architecture

Histoplexer's generator is a U-Net, with 7 levels of downsamplers and 5 levels of upsamplers. The first two layers of downsamplers are optional. When `use_high_res` is set to `true` in `config.json`, the first two layers of downsamplers are turned on. So, the output image's dimension is 4x less than the input on each side.

The main function for running inference on WSI is `get_wsi_inference` in `src/inference/histoplexer_inference_wsi.py`.

`seg_level` in `HistoplexerInferenceWSI` only affects the segmentation step. Using the most high res WSI for segmentation is very slow, so downsampling the WSI is preferred. The downsampling happens in `get_wsi_inference` -> `get_chunks` -> `create_tissue_mask`. This downsampling does not affect any other operation. After the tissue mask is created, it is scaled to the highest resolution of the input WSI, so all coordinates in the tissue mask are relative to the highest resolution.

After tissue mask is created and scaled, image tiles (called "chunks") that overlap with the tissue mask are extracted in `get_wsi_inference` -> `get_chunks` -> `create_tiles_in_mask`. The image tiles are nonoverlapping `shapely.geometry.Polygon` sqaures that store the coordinates of the upper left and lower right corners of the image tiles. The coordinates are also relative to the highest WSI resolution.

A QC image is drawn to show selected image tiles overlapped with the WSI.

The selected image tiles are then passed through the model. For each batch, the model outputs a `list` of 3 tensors at 4x, 16x, and 64x downsampling (relative to highest WSI resolution).

At current setting (2 workers for dataloader, batch size = 128, image size = (48k, 30k)), it takes about 10 GB of system RAM and 14 GB of VRAM. Inference for one WSI takes about 32 seconds.
