"""Consumer-local scratch; exported reference directories remain immutable."""
import pytest


@pytest.fixture(scope='session')
def ppt_cache(tmp_path_factory):
    # Only the filesystem location changes; physical table parameters stay exact.
    return str(tmp_path_factory.mktemp('ppt-cache'))
