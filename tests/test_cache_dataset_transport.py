"""Dataset delivery must preserve the immutable sharded runner's input contract."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_sharded_notebooks as BUILDER
from test_sharded_fallback import fixture_plan


def fixture(base):
    source, global_root = base / "dataset", base / "global"
    source.mkdir(); (global_root / "cache_receipts/shard_0/port").mkdir(parents=True)
    archive = source / "cache384_train.zip.bin"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("cache384_train/synthetic.npy", b"synthetic small ZIP")
    fingerprint = {"plan_sha256": "plan_pin", "archive_sha256": BUILDER.stream_sha256(archive)}
    small = {"cache_summary.json": json.dumps({"storage_bytes": archive.stat().st_size}).encode(),
             "port/plan.csv": b"synthetic header\n"}
    members = []
    for name, data in small.items():
        (global_root / "cache_receipts/shard_0" / name).write_bytes(data)
        members.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    receipts = source / "cache_receipts.tar.gz.bin"
    with tarfile.open(receipts, "w:gz") as tar:
        for name, data in small.items():
            row = tarfile.TarInfo(name); row.size = len(data); tar.addfile(row, io.BytesIO(data))
    dataset_id = "owner/cache-0-bin-v1"
    manifest = {"schema": "rsna_knee_cache_dataset_mirror_v1", "shard": 0,
                "dataset": {"identifier": dataset_id, "version_number": 1},
                "source": {"kernel": "owner/source-cache-0-zip", "version_number": 1},
                "fingerprint": fingerprint,
                "archive": {"filename": archive.name, "original_name": "cache384_train.zip",
                            "bytes": archive.stat().st_size, "sha256": BUILDER.stream_sha256(archive)},
                "receipt_archive": {"filename": receipts.name, "bytes": receipts.stat().st_size,
                                    "sha256": BUILDER.stream_sha256(receipts)}, "receipt_members": members}
    manifest_file = source / "cache_dataset_manifest.json"
    manifest_file.write_text(json.dumps(manifest))
    global_manifest = {"source_plan": [{"shard": 0, "kernel_id": "owner/source-cache-0-zip", "plan_sha256": "plan_pin"}],
                       "cache_input_fingerprints": [fingerprint],
                       "members": [{**row, "path": "cache_receipts/shard_0/" + row["path"]} for row in members]}
    (global_root / "global_metadata_manifest.json").write_text(json.dumps(global_manifest))
    spec = {"dataset_id": dataset_id, "manifest_sha256": BUILDER.stream_sha256(manifest_file), "manifest": manifest, "shard": 0}
    return source, global_root, global_manifest, spec, small


def unpack_fixture(archive, target, expected, allowed):
    if archive.stat().st_size != expected["bytes"] or BUILDER.stream_sha256(archive) != expected["sha256"]:
        raise RuntimeError("Fixture receipt archive SHA mismatch")
    target.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        if {member.name for member in tar.getmembers()} != set(allowed):
            raise RuntimeError("Fixture receipt membership mismatch")
        tar.extractall(target, filter="data")


class CacheDatasetTransportContracts(unittest.TestCase):
    def test_mirror_mount_preserves_receipt_bytes_and_symlinks_archive_without_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, global_root, manifest, spec, small = fixture(base)
            events = []
            target, receipt = BUILDER.prepare_cache_dataset(source, base / "scratch/shard_0", spec, manifest,
                global_root, unpack_fixture, events.append, time.monotonic() + 10)
            for name, data in small.items():
                self.assertEqual((target / name).read_bytes(), data)
            self.assertTrue((target / "cache384_train.zip").is_symlink())
            self.assertEqual((target / "cache384_train.zip").resolve(), (source / "cache384_train.zip.bin").resolve())
            self.assertEqual((target / "cache384_train.zip").stat().st_size, spec["manifest"]["archive"]["bytes"])
            self.assertEqual(events, ["dataset_cache_full_archive_hash", "dataset_cache_receipts_unpack"])
            self.assertGreaterEqual(receipt["archive_hash_seconds"], 0)
            self.assertEqual(receipt["manifest_sha256"], spec["manifest_sha256"])
            self.assertEqual(receipt["dataset_version_number"], 1)

    def test_archive_tamper_and_wrong_source_or_global_fingerprint_fail_before_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, global_root, manifest, spec, _ = fixture(base)
            archive = source / "cache384_train.zip.bin"
            data = bytearray(archive.read_bytes()); data[-1] ^= 1; archive.write_bytes(data)
            with self.assertRaisesRegex(RuntimeError, "archive SHA"):
                BUILDER.prepare_cache_dataset(source, base / "scratch", spec, manifest, global_root,
                    unpack_fixture, lambda _: None, time.monotonic() + 10)
            self.assertFalse((base / "scratch").exists())
            wrong = copy.deepcopy(spec["manifest"]); wrong["source"]["kernel"] = "owner/other-cache"
            with self.assertRaisesRegex(RuntimeError, "source kernel/version"):
                BUILDER.validate_cache_dataset_manifest(wrong, spec["dataset_id"], manifest, global_root)
            wrong = copy.deepcopy(spec["manifest"]); wrong["fingerprint"]["plan_sha256"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                BUILDER.validate_cache_dataset_manifest(wrong, spec["dataset_id"], manifest, global_root)

    def test_receipt_content_changed_even_with_new_archive_pin_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, global_root, manifest, spec, small = fixture(base)
            receipts = source / "cache_receipts.tar.gz.bin"
            with tarfile.open(receipts, "w:gz") as tar:
                for name, data in small.items():
                    if name == "port/plan.csv": data = b"tampered content\n"
                    row = tarfile.TarInfo(name); row.size = len(data); tar.addfile(row, io.BytesIO(data))
            spec["manifest"]["receipt_archive"].update(bytes=receipts.stat().st_size, sha256=BUILDER.stream_sha256(receipts))
            (source / "cache_dataset_manifest.json").write_text(json.dumps(spec["manifest"]))
            spec["manifest_sha256"] = BUILDER.stream_sha256(source / "cache_dataset_manifest.json")
            with self.assertRaisesRegex(RuntimeError, "before copy"):
                BUILDER.prepare_cache_dataset(source, base / "scratch/shard_0", spec, manifest, global_root,
                    unpack_fixture, lambda _: None, time.monotonic() + 10)
            self.assertFalse((base / "scratch/shard_0").exists())

    def test_archive_hash_deadline_refuses_expired_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blob"; path.write_bytes(b"tiny")
            with patch.object(BUILDER.time, "monotonic", return_value=10):
                with self.assertRaisesRegex(RuntimeError, "exhausted Python-stage budget"):
                    BUILDER.stream_sha256(path, deadline=9)

    def test_dataset_scratch_cleanup_covers_prechild_and_second_preparation_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, global_root, manifest, spec, _ = fixture(base)
            working = base / "working"; working.mkdir()
            bad_source = base / "bad_dataset"; bad_source.mkdir()
            (bad_source / "cache_dataset_manifest.json").write_text("wrong version manifest")
            for failure in ("prechild", "second_shard"):
                with self.subTest(failure=failure):
                    namespace = {"Path": Path, "prepare_cache_dataset": BUILDER.prepare_cache_dataset,
                                 "source": source, "bad_source": bad_source, "spec": spec, "manifest": manifest,
                                 "global_root": global_root, "unpack": unpack_fixture, "time": time}
                    body = """import tempfile
_RSNA_CACHE_DATASET_SCRATCH=Path(tempfile.mkdtemp(prefix='rsna-transport-test-',dir='/tmp'))
first,receipt=prepare_cache_dataset(source,_RSNA_CACHE_DATASET_SCRATCH/'shard_0',spec,manifest,global_root,unpack,lambda _:None,time.monotonic()+10)
assert (first/'cache384_train.zip').is_symlink()
"""
                    body += ("raise RuntimeError('Injected GPU selection failure before child')\n" if failure == "prechild" else
                             "prepare_cache_dataset(bad_source,_RSNA_CACHE_DATASET_SCRATCH/'shard_1',spec,manifest,global_root,unpack,lambda _:None,time.monotonic()+10)\n")
                    with self.assertRaisesRegex(RuntimeError, "GPU selection failure|version manifest SHA"):
                        exec(BUILDER.dataset_scratch_guard(body), namespace)
                    self.assertFalse(namespace["_RSNA_CACHE_DATASET_SCRATCH"].exists())
                    self.assertFalse(any(path.is_symlink() for path in working.rglob("*")))
                    self.assertFalse(any(path.name.startswith("cache384_train.zip") for path in working.rglob("*")))
                    self.assertTrue((source / "cache384_train.zip.bin").is_file())

    def test_dataset_builder_removes_cache_kernel_mounts_and_references_distinct_output_slugs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, global_root, _, spec, _ = fixture(base)
            plan, _, _ = fixture_plan()
            plan["kernel_by_shard"] = {shard: f"owner/source-cache-{shard}-zip" for shard in range(7)}
            assets = base / "asset_manifest.json"; assets.write_text("{}")
            docker = base / "docker.json"; docker.write_text(json.dumps({"docker_image": "pinned_docker"}))
            map_file = base / "map.json"; map_file.write_text("{}")
            for mode in ("pilot", "extract", "merge-train"):
                argv = ["builder", "--stage", mode, "--global-root", str(global_root), "--assets-manifest", str(assets),
                        "--docker-metadata", str(docker), "--owner", "owner", "--cache-transport", "datasets",
                        "--cache-dataset-map", str(map_file), "--output-root", str(base / "output")]
                with patch.object(sys, "argv", argv), patch.object(BUILDER.RUNNER, "validate_global", return_value=plan), \
                        patch.object(BUILDER, "load_cache_dataset_map", return_value={0: spec}), contextlib.redirect_stdout(io.StringIO()):
                    BUILDER.main()
            folders = sorted((base / "output").iterdir())
            self.assertEqual({folder.name for folder in folders}, {"rsna-knee-ortho-dataset-pilot", "rsna-knee-ortho-dataset-shard-0", "rsna-knee-ortho-dataset-merge-heads"})
            for folder in folders:
                metadata = json.loads((folder / "kernel-metadata.json").read_text())
                settings = json.loads((folder / "stage_config.json").read_text())
                self.assertEqual(BUILDER.title_for_slug(folder.name), metadata["title"])
                self.assertEqual(metadata["id"].split("/")[-1], folder.name)
                self.assertEqual(metadata["competition_sources"], [])
                self.assertEqual(metadata["docker_image"], "pinned_docker")
                self.assertFalse(any("source-cache" in identifier for identifier in metadata["kernel_sources"]))
                if settings["mode"] == "merge-train":
                    self.assertEqual(metadata["kernel_sources"], [f"owner/rsna-knee-ortho-dataset-shard-{shard}" for shard in range(7)])
                    self.assertEqual(settings["max_seconds"], 1500)
                else:
                    self.assertEqual(metadata["kernel_sources"], [])
                    self.assertIn(spec["dataset_id"], metadata["dataset_sources"])
                    self.assertEqual(settings["cache_dataset_specs"], [spec])
            slugs = ["rsna-knee-ortho-dataset-pilot", *[f"rsna-knee-ortho-dataset-shard-{shard}" for shard in range(7)], "rsna-knee-ortho-dataset-merge-heads"]
            self.assertEqual(len(set(BUILDER.title_for_slug(slug) for slug in slugs)), 9)


if __name__ == "__main__":
    unittest.main()
