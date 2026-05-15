python frankapanda/calibration/stereo.py \
  --depth-backend foundationstereo \
  --foundationstereo-root ~/FoundationStereo \
  --foundationstereo-ckpt ~/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
  --scale 0.5 \
  --valid-iters 16 \
  --max-depth 5
