"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution('espnet').version
except Exception:
    __version__ = '(Not installed from setup.py)'
del pkg_resources
