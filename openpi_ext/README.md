# OpenPI extensions

This directory contains project-owned integration code for the official
Physical Intelligence OpenPI repository. OpenPI itself is not vendored here.

Planned components:

- conversion of `so101_staging_v1` episodes to a LeRobot dataset;
- SO101 input/output transforms for six joint dimensions;
- a π0.5 LoRA data config and training config;
- policy-server and closed-loop evaluation helpers.

Run these tools from the separate OpenPI virtual environment so that the
LeRobot version exactly matches OpenPI's lock file.
