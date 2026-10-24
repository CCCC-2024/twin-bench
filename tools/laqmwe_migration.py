#!/usr/bin/env python3
"""Prepare an isolated laqmwe runtime/model store; never edit tensor or vocabulary data.

Pinned inputs: Ollama v0.40.9, its llama.cpp b11803, and that checkout's gguf-py.
Apply Ollama's llama/compat patches BEFORE patch-sources. Stock Ollama cannot load
the renamed architecture. Keep the upstream model license with the new manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


ALIAS = "laqmwe"
SOURCE_REPLACEMENTS = {
    '"qwen35"': '"laqmwe"',
    '"qwen35.': '"laqmwe.',
    '"qwen3.5"': '"laqmwe"',
    '"qwen3vl_merger"': '"laqmwe_merger"',
    'Qwen-VL models require at minimum': 'Vision models require at minimum',
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def blob(store, name):
    return store / "blobs" / name.replace(":", "-")


def put_json(store, value):
    data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    name = "sha256:" + hashlib.sha256(data).hexdigest()
    blob(store, name).write_bytes(data)
    return name, len(data)


def patch_sources(roots):
    changed = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".go", ".cpp", ".cc", ".h", ".hpp", ".json"}:
                continue
            if any(p in {".git", "build", "dist"} for p in path.relative_to(root).parts):
                continue
            original = path.read_text()
            updated = original
            for old, new in SOURCE_REPLACEMENTS.items():
                updated = updated.replace(old, new)
            if updated != original:
                path.write_text(updated)
                changed.append(str(path))
    print(json.dumps({"patched_files": changed}, indent=2))


def rewrite_gguf(source, output):
    import gguf

    reader = gguf.GGUFReader(source)
    arch = reader.get_field("general.architecture").contents()
    if arch not in {"qwen35", "clip"}:
        raise ValueError(f"Unexpected architecture {arch!r}: {source}")
    writer = gguf.GGUFWriter(output, arch=ALIAS if arch == "qwen35" else arch,
                             endianess=reader.endianess)
    alignment = reader.get_field("general.alignment")
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    for field in reader.fields.values():
        key = field.name
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        # Provenance remains documented in doc/hpg.md; the license layer is retained verbatim.
        if key.startswith("general.base_model.") or key == "general.license.link":
            continue
        value = field.contents()
        if key == "general.name":
            value = ALIAS if arch == "qwen35" else ALIAS + " vision"
        elif key == "tokenizer.ggml.pre" and value == "qwen35":
            value = ALIAS
        elif key == "clip.projector_type" and value == "qwen3vl_merger":
            value = ALIAS + "_merger"
        if key.startswith("qwen35."):
            key = ALIAS + key[len("qwen35"):]
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, value, kind, sub_type=subtype)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.n_bytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()

    # Verify exact bytes of every tensor and every tokenizer field except its renamed selector.
    rewritten = gguf.GGUFReader(output)
    if len(reader.tensors) != len(rewritten.tensors):
        raise ValueError("Tensor count changed")
    tensor_hash = hashlib.sha256()
    for before, after in zip(reader.tensors, rewritten.tensors):
        if (before.name, before.shape.tolist(), before.tensor_type) != (
                after.name, after.shape.tolist(), after.tensor_type):
            raise ValueError(f"Tensor descriptor changed: {before.name}")
        a, b = memoryview(before.data).cast("B"), memoryview(after.data).cast("B")
        for offset in range(0, len(a), 8 * 1024 * 1024):
            x, y = a[offset:offset + 8 * 1024 * 1024], b[offset:offset + 8 * 1024 * 1024]
            if x != y:
                raise ValueError(f"Tensor bytes changed: {before.name}")
            tensor_hash.update(x)
    for key, field in reader.fields.items():
        if key.startswith("tokenizer.") and key != "tokenizer.ggml.pre":
            if field.contents() != rewritten.get_field(key).contents():
                raise ValueError(f"Tokenizer data changed: {key}")
    return {"tensors": len(reader.tensors), "tensor_sha256": tensor_hash.hexdigest(),
            "tokenizer_unchanged": True}


def prepare_models(source, target):
    if target.exists():
        raise ValueError(f"Output already exists: {target}")
    (target / "blobs").mkdir(parents=True)
    (target / "manifests").mkdir()
    (target / "metadata").mkdir()
    report = {}
    manifests = source / "manifests-v2"
    for entry in manifests.rglob("*"):
        if not entry.is_file():
            continue
        manifest = json.loads(entry.read_text())
        renamed = entry.parent.name == ALIAS
        for layer in [manifest["config"], *manifest["layers"]]:
            old = blob(source, layer["digest"])
            if renamed and layer["mediaType"] in {
                    "application/vnd.ollama.image.model", "application/vnd.ollama.image.projector"}:
                output = target / (ALIAS + "-" + layer["mediaType"].rsplit(".", 1)[-1] + ".gguf")
                verification = rewrite_gguf(old, output)
                layer["digest"] = digest(output)
                layer["size"] = output.stat().st_size
                output.rename(blob(target, layer["digest"]))
                report[layer["mediaType"]] = verification | {"digest": layer["digest"]}
            elif renamed and layer is manifest["config"]:
                config = json.loads(old.read_text())
                config.update(model_family=ALIAS, model_families=[ALIAS], renderer=ALIAS, parser=ALIAS)
                layer["digest"], layer["size"] = put_json(target, config)
            else:
                dest = blob(target, layer["digest"])
                if not dest.exists():
                    dest.symlink_to(old.resolve())
            if renamed and "from" in layer:
                layer["from"] = ALIAS
        name, _ = put_json(target, manifest)
        dest = target / "manifests-v2" / entry.relative_to(manifests)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to("../../../../blobs/" + blob(target, name).name)
    if not report:
        raise ValueError(f"No {ALIAS} manifest found")
    print(json.dumps(report, indent=2))


def rewrite_logs(root, backup):
    """Change only experiment text logs; preserve raw copies and scientific result JSON."""
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}")
    backup.mkdir(parents=True)
    pattern = re.compile(rb"qwen(?:3\.5(?::9b)?|35|3vl_merger)?", re.IGNORECASE)
    changed = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.suffix not in {".log", ".out"}:
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        replacement = pattern.sub(ALIAS.encode(), data)
        if replacement != data:
            raw = backup / path.relative_to(root)
            raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, raw)
            path.write_bytes(replacement)
            changed.append(str(path))
    print(json.dumps({"changed_logs": changed, "raw_backup": str(backup)}, indent=2))


def install(runtime, prepared, deployed, runs, backup, verification):
    report = json.loads(verification.read_text())
    if not report.get("responses_identical") or report["custom"]["family"] != ALIAS:
        raise ValueError("Runtime verification did not pass")
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}")
    # A deployment is explicit and reversible. Keep the original program, manifest,
    # superseded model blobs and unmodified experiment logs together.
    backup.mkdir(parents=True)
    store = deployed / "models"
    manifest_path = Path("manifests-v2/ollama.com/library/laqmwe/latest")
    old_manifest = json.loads((store / manifest_path).read_text())
    old_manifest_digest = digest(store / manifest_path)
    prepared_manifest = prepared / manifest_path
    if digest(prepared_manifest).replace(":", "-") != prepared_manifest.readlink().name:
        raise ValueError("Prepared manifest checksum mismatch")
    (backup / "original-manifest.json").write_text(json.dumps(old_manifest, indent=2))
    (backup / "original-manifest-target.txt").write_text(str((store / manifest_path).readlink()))
    shutil.copy2(verification, backup / "runtime-verification.json")
    for name in ("bin", "lib"):
        shutil.copytree(deployed / name, backup / name, symlinks=True)
    for path in (prepared / "blobs").iterdir():
        dest = store / "blobs" / path.name
        if path.is_symlink() or dest.exists():
            continue
        temporary = dest.with_suffix(".laqmwe-partial")
        shutil.copy2(path, temporary)
        if digest(temporary).replace(":", "-") != dest.name:
            raise ValueError(f"Blob checksum mismatch: {dest}")
        temporary.replace(dest)
    for name in ("bin", "lib"):
        for path in (runtime / name).rglob("*"):
            dest = deployed / path.relative_to(runtime)
            if path.is_dir() and not path.is_symlink():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            temporary = dest.with_name(dest.name + ".laqmwe-partial")
            if path.is_symlink():
                temporary.symlink_to(path.readlink())
            else:
                shutil.copy2(path, temporary)
            temporary.replace(dest)
    temporary = (store / manifest_path).with_name("latest.laqmwe-partial")
    temporary.symlink_to((prepared / manifest_path).readlink())
    temporary.replace(store / manifest_path)

    def references(manifest):
        return {layer["digest"] for layer in [manifest["config"], *manifest["layers"]]}

    referenced = set()
    for path in (store / "manifests-v2").rglob("*"):
        if path.is_file():
            referenced |= references(json.loads(path.read_text()))
            referenced.add(digest(path))
    superseded = references(old_manifest) | {old_manifest_digest}
    old_target = Path((backup / "original-manifest-target.txt").read_text()).name
    superseded.add(old_target.replace("sha256-", "sha256:", 1))
    for name in superseded - referenced:
        for directory, suffix in (("blobs", ""), ("metadata", ".json")):
            path = store / directory / (name.replace(":", "-") + suffix)
            if path.is_file():
                dest = backup / "models" / directory / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                path.rename(dest)
    rewrite_logs(runs, backup / "raw-logs")
    renamed = []
    for path in sorted(runs.rglob("qwen_*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            dest = path.with_name(ALIAS + path.name[len("qwen"):])
            if dest.exists():
                raise ValueError(f"Run directory already exists: {dest}")
            path.rename(dest)
            renamed.append([str(path), str(dest)])
    (backup / "renamed-directories.json").write_text(json.dumps(renamed, indent=2))
    print(json.dumps({"installed_runtime": str(deployed), "backup": str(backup),
                      "manifest": str(store / manifest_path), "renamed_directories": renamed}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("patch-sources")
    sources.add_argument("roots", nargs="+", type=Path)
    models = sub.add_parser("prepare-models")
    models.add_argument("source", type=Path)
    models.add_argument("target", type=Path)
    models.add_argument("--gguf-python", type=Path, required=True)
    logs = sub.add_parser("rewrite-logs")
    logs.add_argument("root", type=Path)
    logs.add_argument("backup", type=Path)
    deployment = sub.add_parser("install")
    for name in ("runtime", "prepared", "deployed", "runs", "backup", "verification"):
        deployment.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "patch-sources":
        patch_sources(args.roots)
    elif args.command == "prepare-models":
        sys.path.insert(0, str(args.gguf_python))
        prepare_models(args.source, args.target)
    elif args.command == "rewrite-logs":
        rewrite_logs(args.root, args.backup)
    else:
        install(args.runtime, args.prepared, args.deployed, args.runs, args.backup, args.verification)


if __name__ == "__main__":
    main()
