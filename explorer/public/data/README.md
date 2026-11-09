# Makeup demo

`demo-bundle.json` contains seven makeup tutorials, 124 timed clips, and 30
contexts with their vocabularies and existing annotations. The `transcript`
entries contain timed step descriptions. All media URLs resolve to the
committed files in `../media/makeup/`.

The media are full-length 480p H.264/AAC MP4 demo copies. Their source timing is
preserved for chapter navigation, related clips, and analysis previews.
Tutorial titles and source links are listed below.

- YELLOW CUT CREASE MAKEUP TUTORIAL— PatrickStarrr. [Source video](https://www.youtube.com/watch?v=9f7zmCSzG9E)
- Makeup Tutorial — Smokey Eye Makeup Look + Face & Lips — TheMakeupChair. [Source video](https://www.youtube.com/watch?v=pOMc7gp0KHg)
- 1920s Wearable Makeup Tutorial (Day + Night)  — Jbunzie. [Source video](https://www.youtube.com/watch?v=CMtd-rJcE1k)
- Nude Makeup — Easy Everyday Makeup Tutorial — Indian Skin — corallista. [Source video](https://www.youtube.com/watch?v=GUayJLkqVY8)
- Day to Night Makeup Tutorial for Indian skin — corallista. [Source video](https://www.youtube.com/watch?v=yYOfKGls2Dc)
- TEAL GLITTER SMOKEY EYE (requested by Jaclyn Hill!) — PatrickStarrr. [Source video](https://www.youtube.com/watch?v=pDe_hDjUC74)
- Makeup Tutorial Philippines: Easy Glowy Summer MakeUp Tutorial for 2018 by Dianne — BeKami. [Source video](https://www.youtube.com/watch?v=SsyLv2hCZpw)

Store additional reviewed exports in `local-bundle.json` (ignored by Git), then
set `NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json` and
`NEXT_PUBLIC_DATASET_LABEL=My tutorials` in `.env.local`, and restart the app.
