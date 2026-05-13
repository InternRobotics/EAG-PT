# [Blender]

SCENE_COLLECTION=Blender

SCENE_NAME=kitchen

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}
# # use relighted gt images for evaluation
I_SCENE_EDITING_SCENARIO=100
NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}-relighted

# SCENE_NAME=livingroom

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}
# # use relighted gt images for evaluation
# I_SCENE_EDITING_SCENARIO=110
# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}-relighted



# [FR]

# SCENE_COLLECTION=FR

# SCENE_NAME=classroom

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# # change light colors
# I_SCENE_EDITING_SCENARIO=200
# # turn off and on lights
# I_SCENE_EDITING_SCENARIO=201
# # duplicate chairs
# I_SCENE_EDITING_SCENARIO=202
# # Teaser
# I_SCENE_EDITING_SCENARIO=203

# SCENE_NAME=conference

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0

# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}



# [SelfCapture]

# SCENE_COLLECTION=SelfCaptured

# SCENE_NAME=lectureroom

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}
# use relighted gt images for evaluation
# I_SCENE_EDITING_SCENARIO=301
# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}-relighted-B-front
# # use relighted gt images for evaluation
# I_SCENE_EDITING_SCENARIO=302
# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}-relighted-C-back



# [EFT]

# SCENE_COLLECTION=EFT

# SCENE_NAME=kitchen

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# # insert plane
# I_SCENE_EDITING_SCENARIO=400
# # turn off light and insert light ball
# I_SCENE_EDITING_SCENARIO=401

# SCENE_NAME=furnishedroom

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# # rainbow ceiling light
# I_SCENE_EDITING_SCENARIO=410
# # turn off light and insert light ball
# I_SCENE_EDITING_SCENARIO=411

# SCENE_NAME=emptyroom

# # no scene editing
# I_SCENE_EDITING_SCENARIO=0
# # E-kitchen counter to E-emptyroom
# I_SCENE_EDITING_SCENARIO=420
# # change wall to green
# I_SCENE_EDITING_SCENARIO=421

# NVS_DATASET=_data/${SCENE_COLLECTION}/${SCENE_NAME}



# [---]



EAG_WITH_OPTIMIZED_ALBEDOS_PLY_PATH=_output/${SCENE_COLLECTION}-${SCENE_NAME}_1-diffuse/iter400-plys/optimized-2d-gaussians_iter400.ply

python editing-and-rendering.py \
  --NVS_DATASET_PATH ${NVS_DATASET} --EAG_PLY_PATH ${EAG_WITH_OPTIMIZED_ALBEDOS_PLY_PATH} \
  --I_SCENE_EDITING_SCENARIO ${I_SCENE_EDITING_SCENARIO} \
  --OUTPUT_FOLDER_SUFFIX ${SCENE_COLLECTION}-${SCENE_NAME}_renders_editing-scenario-${I_SCENE_EDITING_SCENARIO}
