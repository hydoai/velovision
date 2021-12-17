# Automated Testing

We use Github Actions to run automated tests.

The desktop development build runs on standard Github Actions runners, but the Jetson deployment build runs on self-hosted runners.

<details>
<summary>How to set up a self-hosted runner</summary>
  
+ Refer to [hydoai/dk1-setup](https://github.com/hydoai/dk1-setup) to set up a Jetson NX for running velovision.
+ From [https://github.com/hydoai/velovision](https://github.com/hydoai/velovision), go to `Settings` -> `Actions` -> `Runners` -> `New self-hosted runner`.
+ Choose Architecture: 'ARM64', then run the shown scripts.
  
</details>
