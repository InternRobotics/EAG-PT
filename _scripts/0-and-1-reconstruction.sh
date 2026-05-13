# [Blender]

SCENE_COLLECTION=Blender
DATASET_IS_SYNTHETIC=t
# [kitchen, livingroom]
SCENE_NAME=kitchen

# [Blender-assets (only needs stage 0 for scene editing)]

# SCENE_COLLECTION=Blender-assets
# DATASET_IS_SYNTHETIC=f
# # [lightball, plane]
# SCENE_NAME=lightball

# [FR]

# SCENE_COLLECTION=FR
# DATASET_IS_SYNTHETIC=f
# # [classroom, conference]
# SCENE_NAME=classroom

# [SelfCaptured]

# SCENE_COLLECTION=SelfCaptured
# DATASET_IS_SYNTHETIC=f
# SCENE_NAME=lectureroom

# [EFT]

# SCENE_COLLECTION=EFT
# DATASET_IS_SYNTHETIC=f
# # [kitchen, furnishedroom, emptyroom]
# SCENE_NAME=furnishedroom



NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}



# [0-radiant-scene-reconstruction.py]

LOSS_NORMAL_SUPERVISION=0.5
LOSS_NORMAL_CONSISTENCY=0.05

python 0-radiant-scene-reconstruction.py \
  --NVS_DATASET_PATH ${NVS_DATASET} --DATASET_IS_SYNTHETIC ${DATASET_IS_SYNTHETIC} \
  --LOSS_NORMAL_SUPERVISION ${LOSS_NORMAL_SUPERVISION} --LOSS_NORMAL_CONSISTENCY ${LOSS_NORMAL_CONSISTENCY} \
  --OUTPUT_FOLDER_SUFFIX ${SCENE_COLLECTION}-${SCENE_NAME}_0-radiant



# [1-diffuse-material-recovery.py]

if [ ${SCENE_COLLECTION} != "Blender-assets" ]; then

FINETUNED_TDGT_PLY_PATH=_output/${SCENE_COLLECTION}-${SCENE_NAME}_0-radiant/iter30000-plys/optimized-2d-gaussians.ply

SPP=256

python 1-diffuse-material-recovery.py \
  --NVS_DATASET_PATH ${NVS_DATASET} --DATASET_IS_SYNTHETIC ${DATASET_IS_SYNTHETIC} \
  --EAG_PLY_PATH ${FINETUNED_TDGT_PLY_PATH} \
  --N_SPP_OPTIMIZE_ALBEDO ${SPP} \
  --OUTPUT_FOLDER_SUFFIX ${SCENE_COLLECTION}-${SCENE_NAME}_1-diffuse

fi
