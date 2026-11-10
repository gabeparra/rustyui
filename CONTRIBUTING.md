# Contributing to rustyui

Thanks for your interest in improving rustyui. Bug reports, fixes, new nodes and new model families
are all welcome.

## Before you start

- For anything larger than a small fix, open an issue first so the approach can be agreed on before
  you spend time on it.
- rustyui's promise is that a graph renders the same here as in ComfyUI. A change to a node, a
  sampler or a kernel should keep that true, and the pull request should say how it was checked.

## Development setup

See the README for requirements. The pinned container in `dev/` (see `dev/README.md`) is the
simplest way to get every native dependency at the right version. Then:

```bash
(cd ui && pnpm install --frozen-lockfile && pnpm exec vite build --config vite.config.mts)
(cd frontend && bun install --frozen-lockfile && bun run build)
cargo build --release
cargo test --release -p rustyui-backend
```

For editor work, run the backend and the Vite dev server side by side:

```bash
./target/release/rustyui-backend serve --models /path/to/models
(cd frontend && bun run dev)     # the simple editor with hot reload; API calls go on to :8190
```

## Checks

CI runs these on every pull request; please run them locally first:

```bash
cargo fmt -p rustyui-backend --check
cargo clippy --release -p rustyui-backend --no-deps -- -D warnings
cargo test --release -p rustyui-backend
(cd frontend && bun run typecheck && bun run build && node --experimental-strip-types --test tests/forge.test.ts)
```

`ui/` is upstream ComfyUI_frontend; changes there should be kept small and listed in
`ui/UPSTREAM.md`, so a later upstream version can be taken without losing them.

## Matching ComfyUI

When you port a node or a model:

- Follow ComfyUI's code path, including dtypes and the order of operations, and cite the ComfyUI
  file the code follows in the module's doc comment.
- Compare against a real ComfyUI render of the same graph. `rustyui-backend compare a.png b.png`
  prints the mean and max pixel difference and the PSNR; include those numbers in the pull request.
- For bit-exact work, capture reference tensors from ComfyUI as the scripts in `backend/refs/` do,
  and add a `debug-*` command that compares against them.

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
