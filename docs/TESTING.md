
# Automated Testing

We use Github Actions to run automated tests.

The desktop development build runs on standard Github Actions runners, but the Jetson deployment build runs on self-hosted runners.


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
# Table of Contents

- [How to set up a self-hosted runner](#how-to-set-up-a-self-hosted-runner)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# How to set up a self-hosted runner

1. Refer to [hydoai/dk1-setup](https://github.com/hydoai/dk1-setup) to set up a Jetson NX for running velovision.
1. From [https://github.com/hydoai/velovision](https://github.com/hydoai/velovision), go to `Settings` -> `Actions` -> `Runners` -> `New self-hosted runner`.
1. Choose Architecture: 'ARM64', then run the shown scripts.
  
