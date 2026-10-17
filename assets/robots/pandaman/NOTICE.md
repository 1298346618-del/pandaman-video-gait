# Pandaman robot description — provenance and license

The files in this directory (`mjcf/`, `urdf/`, `meshes/`) describe the **Pandaman**
10-DoF humanoid robot. They were originally distributed with the Humanoid-Gym /
Humanoid-Gym-Modified ecosystem and originate from the legged_gym–style asset
pipeline maintained by Beijing RobotEra Technology.

They are redistributed here **unchanged**, solely so that the experiments in this
repository are reproducible from a single clone.

## License

The upstream project is distributed under the **BSD-3-Clause** license:

```
Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2021 ETH Zurich, Nikita Rudin
Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The MIT license in the repository root applies to **our** code only, not to these
assets.

## Note on MuJoCo versions

`mjcf/pandaman.xml` uses the legacy `<flag sensornoise="enable">` attribute, accepted
by MuJoCo 2.3.3 (the version used in the paper) but rejected by MuJoCo ≥ 3.1. If you
run a newer MuJoCo, delete **only** the `sensornoise="enable"` token from the `<flag>`
element.

Do **not** remove `frictionloss="enable"` as well: a joint in this model carries
`frictionloss="0.01"`, so dropping that flag silently changes the contact/joint
dynamics and your results will no longer be comparable with the paper. For exact
reproduction use `mujoco==2.3.3`.
