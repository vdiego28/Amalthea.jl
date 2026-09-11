"""Rootless minimum-glibc validation contracts; no network or namespace access."""
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent/'standalone_wheels'))
import glibc228


class GlibcValidationTests(unittest.TestCase):
    def test_architecture_qualified_vendor_package_manifest(self):
        for name in ('libc6', 'libc6:amd64'):
            self.assertEqual(glibc228.libc_package_version(f'base-files\t10\n{name}\t2.28-10+deb10u1\n'),
                             '2.28-10+deb10u1')
        for text in ('libc6:amd64\t2.31-1\n', 'libc6:arm64\t2.28-1\n', ''):
            with self.assertRaises(ValueError):
                glibc228.libc_package_version(text)

    def test_absolute_in_image_symlink_is_preserved_without_host_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root/'image.tar'
            with tarfile.open(archive, 'w') as stream:
                regular = tarfile.TarInfo('usr/lib/value'); regular.size = 4
                stream.addfile(regular, io.BytesIO(b'data'))
                link = tarfile.TarInfo('lib/value'); link.type = tarfile.SYMTYPE
                link.linkname = '/usr/lib/value'; stream.addfile(link)
            destination = root/'image'
            glibc228.extract_rootfs(archive, destination)
            self.assertEqual((destination/'lib/value').read_bytes(), b'data')
            self.assertEqual((destination/'lib/value').resolve(), destination/'usr/lib/value')

    def test_outside_relative_links_traversal_and_devices_are_rejected(self):
        for kind in ('link', 'traversal', 'device'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); archive = root/'image.tar'
                with tarfile.open(archive, 'w') as stream:
                    member = tarfile.TarInfo('../outside' if kind == 'traversal' else 'entry')
                    if kind == 'link':
                        member.type = tarfile.SYMTYPE; member.linkname = '../outside'
                    elif kind == 'device':
                        member.type = tarfile.CHRTYPE
                    stream.addfile(member)
                with self.assertRaises((ValueError, tarfile.FilterError)):
                    glibc228.extract_rootfs(archive, root/'image')
                self.assertFalse((root/'outside').exists())

    def test_container_has_only_explicit_bindings_and_no_network_or_host_environment(self):
        with patch.object(glibc228.shutil, 'which', return_value='/usr/bin/bwrap'):
            command = glibc228.container_command(Path('/tmp/rootfs'), Path('/tmp/python'),
                Path('/tmp/work'), Path('/tmp/evidence'), bindings=[(Path('/tmp/wheels'), '/wheels')])
        for option in ('--clearenv', '--unshare-net', '--unshare-user', '--unshare-pid'):
            self.assertIn(option, command)
        self.assertEqual(command[command.index('--ro-bind')+1:command.index('--ro-bind')+3], ['/tmp/rootfs', '/'])
        self.assertNotIn('/usr', command)
        self.assertNotIn('/home', command)
        self.assertIn('/nonexistent', command)

    def test_no_bubblewrap_cannot_fall_back_to_host_execution(self):
        with patch.object(glibc228.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'host execution is not an alternative'):
                glibc228.container_command(Path('/tmp/rootfs'), Path('/tmp/python'),
                                          Path('/tmp/work'), Path('/tmp/evidence'))


if __name__ == '__main__':
    unittest.main()
