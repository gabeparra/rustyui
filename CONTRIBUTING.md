# Contributing to rustyui

Thanks for your interest in improving rustyui. Bug reports, fixes, new nodes and new model families
are all welcome.

## Where to help

- Issues labelled [`good first issue`](https://github.com/gabeparra/rustyui/labels/good%20first%20issue)
  need no GPU or are small and well bounded; [`help wanted`](https://github.com/gabeparra/rustyui/labels/help%20wanted)
  are larger and open for anyone to take. Say in the issue that you are on it.
- The [roadmap](README.md#limitations-and-roadmap) lists what is planned. A ComfyUI node rustyui
  refuses ("not ported yet") is always a welcome port: one node, checked against ComfyUI, is a
  good-sized pull request.
- A graph that renders differently from ComfyUI is the most useful bug report there is (label
  `exactness`). Attach the graph and both outputs.
- Without an NVIDIA card you can still work on the editor, the HTTP API, the documentation and
  everything the unit tests cover; CI builds the backend and runs those tests for you.
- Questions and ideas go to [Discussions](https://github.com/gabeparra/rustyui/discussions).

## Before you start

- For anything larger than a small fix, open an issue first so the approach can be agreed on before
  you spend time on it.
- rustyui's promise is that a graph renders the same here as in ComfyUI. A change to a node, a
  sampler or a kernel should keep that true, and the pull request should say how it was checked.

## Development setup

See the README for requirements. The pinned container in `dev/` (see `dev/README.md`) is the
simplest way to get every native dependency at the right version. Then:

```bash
(cd frontend && bun install --frozen-lockfile && bun run build)
cargo build --release
cargo test --release -p rustyui-backend
```

For editor work, run the backend and the Vite dev server side by side:

```bash
./target/release/rustyui-backend serve --models /path/to/models
(cd frontend && bun run dev)     # the editor with hot reload; API calls go on to :8190
```

## Checks

CI runs these on every pull request; please run them locally first:

```bash
cargo fmt -p rustyui-backend --check
cargo clippy --release -p rustyui-backend --no-deps -- -D warnings
cargo test --release -p rustyui-backend
(cd frontend && bun run typecheck && bun run build && node --experimental-strip-types --test tests/forge.test.ts)
```

## Pull requests

- Work on a branch of your fork and open the pull request against `main`. Keep one topic to a pull
  request; a node and an unrelated fix are two.
- CI runs on every pull request. For a first contribution a maintainer has to start it, so it may
  wait a little.
- Fill in the template: what changed and how it was checked. For anything that touches rendering
  that means the graph you ran and how the output compared with ComfyUI's; say so plainly if you
  could not check it on a card.
- Pull requests are merged by squash or rebase, so `main` stays a straight line of commits that
  each build. The pull request's title and description become the commit, so write them as one
  (see Style).
- Review comments are meant to get the change in, not to keep it out. If a review asks for a check
  you cannot run, say so and it will be run for you.

## Matching ComfyUI

When you port a node or a model:

- Follow ComfyUI's code path, including dtypes and the order of operations, and cite the ComfyUI
  file the code follows in the module's doc comment.
- Compare against a real ComfyUI render of the same graph. `rustyui-backend compare a.png b.png`
  prints the mean and max pixel difference and the PSNR; include those numbers in the pull request.
- For bit-exact work, capture reference tensors from ComfyUI as the scripts in `backend/refs/` do,
  and add a `debug-*` command that compares against them.
- A change that is meant only to be faster keeps every output value: `rustyui-backend debug-passes`
  compares the fast kernels in `backend/kernels/ops.cu` bit for bit with the forms they replace, and
  a new one belongs in that command.

Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) too: it applies to issues, pull requests and
discussions alike.

## Style

- Rust: `rustfmt` with the repository's settings, no new clippy warnings, `anyhow` errors with
  context a user can act on.
- TypeScript and Vue: strict types, no new dependencies without a reason in the pull request.
- No Python in the program itself. Python belongs only in `backend/refs/`, the scripts that capture
  ComfyUI's tensors to check rustyui against. When ComfyUI's numbers come from a native library,
  link or port that library.
- Comments explain why, not what. Keep them short.
- Commit messages: a specific subject line, then a short paragraph on the reason for the change.

## License

By contributing you agree that your contributions are licensed under the GNU General Public License
v3.0, the license of this project.
