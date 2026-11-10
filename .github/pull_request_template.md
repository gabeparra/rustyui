## What this changes

<!-- A short description, and the issue it closes if there is one. -->

## How it was checked

<!-- For rendering changes: the graph used and the `rustyui-backend compare` numbers against ComfyUI. -->

## Checklist

- [ ] `cargo fmt -p rustyui-backend --check`
- [ ] `cargo clippy --release -p rustyui-backend --no-deps -- -D warnings`
- [ ] `cargo test --release -p rustyui-backend`
- [ ] `cd frontend && bun run typecheck && bun run build`
- [ ] CHANGELOG.md updated under Unreleased, if users would notice the change
