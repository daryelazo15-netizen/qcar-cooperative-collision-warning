# Checkpoints

The weights are not stored in git (81 MB each). They are release assets of the repository; this file fixes what they must be.

| name | role | SHA-256 | size (bytes) |
|---|---|---|---|
| `net_epoch5.pth` | cooperative detector (HEAL pyramid fusion, two cars), selected epoch 5 | `968de39a20385bbe7686101f30cefcd7418cabf9e24030551e10e980d1f88743` | 81159177 |
| `net_epoch11.pth` | single-agent detector (ego car only), selected epoch 11 | `f51ac63317a46f15b8d0aea0fb3bb896826371db565070218d3db819730c360c` | 81159177 |

Place each file in its model directory next to the matching `resolved_hypes.json` (`perception/model_dirs/coop/`, `perception/model_dirs/single/`). The live loop checks the
hash against `_live_expected_checkpoint_sha256` in that file before it starts. To check a download by hand:

    shasum -a 256 net_epoch5.pth net_epoch11.pth

Download: release [`checkpoints-v1`](https://github.com/daryelazo15-netizen/qcar-cooperative-collision-warning/releases/tag/checkpoints-v1) of this repository.
