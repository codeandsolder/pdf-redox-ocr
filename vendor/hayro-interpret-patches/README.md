# Hayro interpreter patches

The sibling `hayro-interpret` tree is generated from the exact upstream
[Hayro](https://github.com/LaurenzV/hayro) revision in `UPSTREAM_REV`.
The two API patches expose resolved raster decoding identities and exact page
XObject invocation paths. Their upstream-ready commits are published in
[Hayro PR #5](https://github.com/codeandsolder/hayro/pull/5) and
[Hayro PR #6](https://github.com/codeandsolder/hayro/pull/6).

The standalone manifest pins the unchanged Hayro siblings directly to the same
upstream revision. OCR's Cargo patch also selects the shared syntax snapshot in
the pinned pdf-redox repository, so the scanner, interpreter and optimizer use
exactly one syntax crate. No Hayro fork is a build dependency.

To regenerate or verify the checked-in snapshot from a local upstream checkout:

```sh
vendor/hayro-interpret-patches/refresh.sh /path/to/hayro
vendor/hayro-interpret-patches/refresh.sh /path/to/hayro --check
```

Make interpreter changes in an upstream checkout, regenerate the patch series
and then refresh the snapshot. Do not hand-edit the generated tree. Remove
each patch when the equivalent API is available upstream; update the upstream
pin, pinned siblings and the shared syntax snapshot together.
