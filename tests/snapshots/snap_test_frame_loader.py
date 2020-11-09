# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots["test_get_all_frames 1"] = (480, 854, 3)

snapshots["test_get_all_frames 2"] = (480, 854, 3)
