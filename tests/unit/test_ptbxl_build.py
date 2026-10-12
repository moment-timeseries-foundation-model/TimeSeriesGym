"""The PTB-XL challenge zip is reproducible.

The same PTB-XL release gives the same zip, h5 and CSV bytes whatever the directory listing
order, file mtimes, permissions or the time of the build.
"""

import hashlib
import os
import zipfile

from timeseriesgym.utils import ZIP_DATE_TIME, compress, extract


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- zip


def test_compress_zip_does_not_depend_on_order_mtimes_or_permissions(tmp_path):
    files = {"b.csv": b"label\n0\n", "sub/a.h5": bytes(range(256)) * 64, "run.sh": b"#!/bin/sh\n"}
    zips = []
    builds = [("one", sorted, 1.0e9, 0o600), ("two", reversed, 1.7e9, 0o664)]
    for name, order, mtime, mode in builds:
        src = tmp_path / name
        for rel in order(sorted(files)):  # created in opposite orders
            path = src / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[rel])
            path.chmod(mode | (0o100 if rel == "run.sh" else 0))
            os.utime(path, (mtime, mtime))
        zips.append(tmp_path / f"{name}.zip")
        compress(src, zips[-1])

    assert _md5(zips[0]) == _md5(zips[1])
    with zipfile.ZipFile(zips[0]) as archive:
        infos = archive.infolist()
    assert [info.filename for info in infos] == ["b.csv", "run.sh", "sub/a.h5"]
    assert {info.date_time for info in infos} == {ZIP_DATE_TIME}
    assert {info.compress_type for info in infos} == {zipfile.ZIP_DEFLATED}
    modes = {info.filename: info.external_attr >> 16 for info in infos}
    assert modes == {"b.csv": 0o100644, "run.sh": 0o100755, "sub/a.h5": 0o100644}

    extract(zips[1], tmp_path / "extracted")
    for rel, data in files.items():
        assert (tmp_path / "extracted" / rel).read_bytes() == data
