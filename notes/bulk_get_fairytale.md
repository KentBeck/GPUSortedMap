In the land of WGPU, there lived a quiet kingdom called the Slab. The Slab kept its keys in perfect order, guarded by a wise keeper named Meta, who always knew the length of the realm.

One day, a traveler brought a list of keys and asked the kingdom for their values. The Queen (our program) prepared three gifts for the journey:

- A scroll of keys for the messengers to read (the keys buffer).
- A blank ledger where answers would be written (the results buffer).
- A tiny note telling how many keys there were (the keys meta).

The Queen summoned a band of sixty-four sprites at a time; each sprite took one key, dashed into the Slab, and performed a swift binary search through the ordered halls. If the key was found, the sprite etched its value into the ledger and marked it as found. If not, it left the page blank.

When all sprites returned, the ledger was carried back across the boundary (readback), and the Queen's scribes translated the markings into a list of answers for the traveler.

And that is how bulk_get journeys through the GPU realm: a scroll of keys, a band of sprites, a swift search, and a ledger of results.
