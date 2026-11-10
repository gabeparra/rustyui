# Security policy

## Supported versions

Security fixes go into the latest release and the `main` branch.

## Design

rustyui is a local, single-user server. It is built to be reachable only from the machine it runs
on:

- it refuses to listen on anything but a loopback address;
- it refuses requests whose `Host` is not a loopback name (DNS rebinding), whose `Origin` is not
  the editor's own, or that the browser marks as cross-site (`Sec-Fetch-Site`);
- every model name, output prefix and `/view` path is kept inside its folder, with symlinks
  resolved.

There is no authentication. Do not expose the port to a network; use an ssh tunnel to reach it
from another machine.

Models are read from `safetensors` and GGUF files. ESRGAN `.pth` upscalers go through candle's
pickle reader, which only decodes tensor data and never runs code from the file.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting instead (the **Security** tab of the repository, then **Report a vulnerability**).

Include what you found, how to reproduce it, and what an attacker could do with it. You should get
an answer within a week. Once a fix is released, the report can be credited in the changelog if
you wish.
