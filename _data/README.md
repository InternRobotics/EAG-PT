# EAG-PT Data

[**EAG-PT Data at Hugging Face**](https://huggingface.co/datasets/XijieYang/EAG-PT/tree/main)

Multi-view indoor scene datasets can be downloaded at [huggingface](https://huggingface.co/datasets/XijieYang/EAG-PT/tree/main) in browser or by [hf cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli). We recommend you first download `Blender.zip` and `Blender-assets.zip` (for scene editing) to test the code. e.g. `hf download --repo-type=dataset XijieYang/EAG-PT Blender-assets.zip Blender.zip --local-dir .`

```sh
_data
├── Blender
│   ├── kitchen
│   ├── kitchen-relighted
│   ├── livingroom
│   └── livingroom-relighted
├── Blender-assets
│   ├── lightball
│   └── plane
├── EFT
│   ├── emptyroom
│   ├── furnishedroom
│   └── kitchen
├── FR
│   ├── classroom
│   └── conference
└── SelfCaptured
    ├── lectureroom
    ├── lectureroom-relighted-B-front
    └── lectureroom-relighted-C-back
```

- We render `Blender` and `Blender-assets` in Blender using scripts.
- FR data are re-organized from [FIPT](https://github.com/Jerrypiglet/rui-indoorinv-data/tree/fipt).
- `SelfCaptured` is captured and processed by ourselves.
- EFT data are converted from [VR-NeRF](https://github.com/facebookresearch/EyefulTower/tree/main).
