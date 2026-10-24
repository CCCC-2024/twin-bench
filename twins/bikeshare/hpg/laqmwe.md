# The laqmwe deployment

`laqmwe` is a local deployment of Qwen3.5-9B with renamed metadata and a matching native runtime. It is not a new
trained model. Results published from this deployment should identify the original model.

The runtime is based on Ollama `v0.40.9` and its pinned llama.cpp `b11803`. The changes select the original model
implementation using `laqmwe` as the architecture, tokenizer selector, renderer and parser; the vision projector
uses `laqmwe_merger`. One vision-loader message uses a generic description. This is a compiled runtime change,
not a CLI output filter. Stock Ollama cannot load the renamed GGUF files.

The GGUF rewrite preserves all tensor names, types, shapes and bytes, the vocabulary, merges, special token IDs,
chat template and upstream license. Display names and architecture selectors are changed; optional base-model
display metadata and the redundant license URL are omitted. The original model identity remains documented here.
Manifests and blob filenames use recalculated SHA-256 digests.

## Reproduce

Use a GPU allocation for inference. Build and download dependencies in a node-local directory, not the home cache.
Go `1.26.10`, CMake `3.24+`, a C++17 compiler and the checkout's `gguf-py` dependencies are required. The original
model store must already contain the `laqmwe` alias. The migration tool does not download model weights.

1. Download and extract [Ollama v0.40.9](https://github.com/ollama/ollama/releases/tag/v0.40.9) and
   [llama.cpp b11803](https://github.com/ggml-org/llama.cpp/releases/tag/b11803).
   Archive SHA-256 values used for this build:
   - Ollama: `9e99fd4c86ebb5ff948a7f22a667b0c0a5d80a14b86aaf232f282c092e0d6ca7`
   - llama.cpp: `fff546943c8ba3169c1c721a961e52595d4eac0e35b0c5ad768eb33ad846eba8`
2. Apply Ollama's `llama/compat` patches to llama.cpp, following its `cmake/apply-git-patches.cmake`. Then run
   `python tools/laqmwe_migration.py patch-sources <ollama-source> <llama-source>`.
3. Build the native payload from `<ollama-source>/llama/server`, with the patched llama.cpp supplied through
   `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP` and `OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON`. Build and install `llama-server`
   into an isolated runtime. Build the Go binary from the patched Ollama checkout. Keep the matching CUDA libraries
   and license files from the original v0.40.9 installation with that runtime.
4. Create a separate model store:
   ```sh
   python tools/laqmwe_migration.py prepare-models <original-store> <new-store> --gguf-python <llama-source>/gguf-py
   ```
   The command rejects an existing output directory. It verifies every tensor byte and tokenizer field after
   writing. Unmodified layers in the temporary store link to the original store.
5. Validate both runtimes against the original model and the rewritten model:
   ```sh
   python tools/check_laqmwe_runtime.py --stock-runtime <original-runtime> --stock-store <original-store> \
     --runtime <new-runtime> --store <new-store> --report <report.json>
   ```
   The check uses isolated localhost ports 12671/12672 and stops both servers afterwards. It checks native
   `list`, `show`, `show --modelfile/--parameters/--template/--license`, `ps`, the default `/api/show`, and loader logs.
   Text, a red test image and the OpenAI-compatible chat API must work and produce the same answers as stock.
   Native process arguments must use the new name, and gpt-oss inspection and inference must still work.

`install` is the explicit deployment step after verification. It backs up the runtime and original model manifest,
installs verified blobs and programs, switches the local manifest, and moves superseded unreferenced model blobs
and metadata into the backup. Experiment `.log`/`.out` files use the new name, with raw originals preserved in the
backup; `qwen_*` run directories become `laqmwe_*`. Scientific result JSON and repository source snapshots retain
their original identities. Do not run deployment while an Ollama service is using the installation.
The experiment scripts print summaries for the models requested in the current invocation, so an old model name
retained in the result JSON does not reappear in a new run's summary.

## Limits and recovery

Renaming cannot make the model's origin unidentifiable. Original backups, scientific records, model structure and
weight fingerprints remain. Other model families and upstream implementation symbols in the runtime retain their
names. A complete case-insensitive byte scan also finds incidental `qwen` sequences in the unchanged quantized
tensor payload. Removing those sequences would alter weights; `strings` is therefore not guaranteed to return no
matches. Cluster-owned audit records are outside this migration.

Do not overwrite this runtime with a stock Ollama update while the renamed model is active. `setup.sh` reuses an
installed runtime. To revert, stop its services, restore `bin/` and `lib/` from the deployment backup, restore the
original blobs and metadata from `models/`, and repoint the `laqmwe/latest` manifest using
`original-manifest-target.txt`. `raw-logs/` and `renamed-directories.json` record the original experiment files.
