// rustyui: TORCH_CHECK for the vendored FMHA forward, whose only use of it is the host-side
// check_supported(), a bool: a failed check returns false there.
#pragma once
#define TORCH_CHECK(cond, ...) \
  do {                         \
    if (!(cond)) return false; \
  } while (0)
