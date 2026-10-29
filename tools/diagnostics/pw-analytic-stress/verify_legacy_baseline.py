"""Verify the independent accepted joint-mixer baseline before any compilation."""
from pathlib import Path
import hashlib
import json

ARCHIVE_SHA256 = '7c8f4f9d90253a3fb92d57cbf124dbea0fb60654057f239e5af24838b76a34c2'
SOURCE_SHA256 = 'c4820e889075ec04217d5296e8a96039f6fc77e674a1646db1578049b1b00008'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(archive, include_root, source_manifest, candidate_archive):
    archive, include_root = Path(archive), Path(include_root)
    source_manifest, candidate_archive = Path(source_manifest), Path(candidate_archive)
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('legacy archive is not the accepted 1480 current-joint baseline')
    if digest(source_manifest) != SOURCE_SHA256:
        raise ValueError('legacy header manifest is not the accepted 1480 source')
    with candidate_archive.open('rb') as stream:
        if stream.read(8) != b'!<arch>\n':
            raise ValueError('candidate is not a static archive')
    candidate_sha = digest(candidate_archive)
    if candidate_sha == ARCHIVE_SHA256:
        raise ValueError('legacy baseline and candidate must be independent archive bytes')
    expected = {n[len('include/'):]: h for n, h in
                json.loads(source_manifest.read_text()).items() if n.startswith('include/')}
    actual = {str(p.relative_to(include_root)): digest(p)
              for p in include_root.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('legacy include root differs from the complete accepted header closure')
    return {'baseline_archive_sha256': ARCHIVE_SHA256,
            'baseline_source_manifest_sha256': SOURCE_SHA256,
            'baseline_header_files': len(expected),
            'candidate_archive_sha256': candidate_sha,
            'candidate_distinct': True,
            'scope': 'Baseline identity only; candidate native numerical qualification is separate.'}
