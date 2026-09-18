# samples/

Demo clips for the "Not posted yet — upload file" mode and for the startup warmup are not
bundled with this repository: the clips used during development are real TikTok videos from the
`lingbow/tiktok-video-engagement-200k` dataset (CC BY-NC 4.0, see `data/README.md`) and are not
redistributed here.

To enable the demo button locally, put a clip named `demo1_hit.mp4` into this folder and describe
it in `manifest.json`:

```json
[{"name": "demo1_hit", "label": "hit", "caption": "…", "topic": "…", "duration": 67, "post_time": "2024-07-06T20:27"}]
```

Without a clip the app works as usual; the warmup skips the Whisper and SigLIP/CLAP steps and the
demo button reports that no sample is available.
