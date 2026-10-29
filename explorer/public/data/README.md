# Example data

`demo-bundle.json` contains four entirely synthetic tutorials, sixteen knowledge items,
and two contextual units (`paper`, `stage`). It contains no real participants,
study responses, third-party transcripts, or model request logs.

`../media/folding-demo.mp4` is an original generated title card for local playback.
It illustrates seeking and timeline interactions; it is not a recorded folding tutorial.
All synthetic example text and media in this directory are dedicated to the public
domain under CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/

Put your own reviewed exports in `local-bundle.json` (ignored by Git), then set
`NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json` in `.env.local` and restart Next.js.
