# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node that mimics policy server for testing."""

import dora
import numpy as np
import pyarrow as pa


def _generate_dummy_actions():
    positions = []
    base = np.array(
        [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    for _ in range(10):
        noise = np.random.uniform(-0.3, 0.3, size=base.shape).astype(np.float32)
        positions.append((base + noise).tolist())
    return {
        "interval": int(1e9 / 30),
        "positions": positions,
    }


def main():
    """Mimics policy server."""
    node = dora.Node()
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        actions = _generate_dummy_actions()
        node.send_output(
            "actions",
            pa.array(actions["positions"], type=pa.list_(pa.float32())),
            {"interval": actions["interval"]},
        )


if __name__ == "__main__":
    main()
