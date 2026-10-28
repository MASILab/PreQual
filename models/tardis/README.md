# TARDIS model checkpoints

Place the two TARDIS rectified-flow checkpoints here before building the container:

    models/tardis/flow_difov_2d_basegd_multiB_best.pt    FOD model (30-direction, multi-b)
    models/tardis/flow_difov_2d_multicond_t1_best.pt     tensor model (6-direction)

The Singularity recipe copies them to `/APPS/tardis/checkpoints/` with `%files`; the build fails if
either file is missing. The `.pt` files are gitignored.
