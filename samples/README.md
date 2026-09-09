# Sample images

Shared by both apps in this repo: the public page (`site/`) and the dev
playground (`playground/`). They live here, at the top level, so neither app
reaches into the other's assets.

## `cafe-table.jpg`

The photo both apps preload first. Chosen for its several clearly separable
objects — croissant, wooden plate, fork, spoon, glass of berries, pine cones, a
paper tag and a hand — which is what makes an everything-mode segmentation
result readable at a glance.

- **Source:** <https://commons.wikimedia.org/wiki/File:Rustic_Cafe_Table_(Unsplash).jpg>
- **Originally published:** <https://unsplash.com/photos/L_ENUhk011o>
- **Author:** Kawin Harasai
- **Licence:** CC0 1.0 Universal (Public Domain Dedication) —
  <https://creativecommons.org/publicdomain/zero/1.0/>
- **Modifications:** downscaled from 4018x2548 to 1024x649 and re-encoded as
  JPEG quality 45, to keep the committed file under 200 KB.

## `sticky-notes.jpg`

The public page's second sample: a visitor trying a *note* scanner should be
able to segment notes. Roughly sixty well-separated, high-contrast sticky notes
on glass, no identifiable faces, generic workshop content. Note density matters
— the page runs at `pointsPerSide: 16` (256 sample points), and this photo's
note count sits in the range that grid can actually resolve.

- **Source:** <https://commons.wikimedia.org/wiki/File:Off_the_wall_ideas_sticky_notes_SEI_2018_(41779807090).jpg>
- **Originally published:** <https://www.flickr.com/photos/156788110@N04/41779807090>
- **Author:** embljusocmedia (U.S. Embassy in Ljubljana Flickr stream), 2018-07-23
- **Licence:** Public domain as a work of the U.S. Department of State
  (`{{PD-USGov-DOS}}`) —
  <https://commons.wikimedia.org/wiki/Template:PD-USGov-DOS>. Independently
  Flickr-reviewed by `FlickreviewR 2` on 2024-08-16, review licence "Public
  Domain Mark". The basis is the U.S. Government authorship, not a licence tag
  an uploader applied to someone else's photo.
- **Modifications:** fetched at `width=800` via the Commons `Special:FilePath`
  renderer; otherwise unmodified.

CC0 and public domain waive copyright, so no attribution is legally required.
It is recorded here anyway: a committed binary whose provenance nobody can
state is a binary nobody can safely publish — and these two are going on a
public web page. Neither file is shipped in the npm tarball; `package.json`
sets `"files": ["dist"]`.
