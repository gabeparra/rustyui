//! RUSTYUI_PROFILE=1: time named sections, synchronising the card around each one so the
//! numbers are real GPU time. Off by default (no synchronisation, no cost).

use candle_core::{Device, Result};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::time::Instant;

thread_local! {
    static ON: bool = std::env::var_os("RUSTYUI_PROFILE").is_some();
    static TIMES: RefCell<BTreeMap<String, (f64, u64)>> = const { RefCell::new(BTreeMap::new()) };
    static SCOPE: Cell<&'static str> = const { Cell::new("") };
}

/// Prefix for the sections timed from now on ("unet", "vae", "clip").
pub fn scope(s: &'static str) {
    SCOPE.with(|c| c.set(s));
}

pub fn enabled() -> bool {
    ON.with(|o| *o)
}

pub fn time<R>(name: &'static str, dev: &Device, f: impl FnOnce() -> Result<R>) -> Result<R> {
    if !enabled() {
        return f();
    }
    dev.synchronize()?;
    let t = Instant::now();
    let r = f()?;
    dev.synchronize()?;
    let secs = t.elapsed().as_secs_f64();
    TIMES.with(|m| {
        let mut m = m.borrow_mut();
        let key = SCOPE.with(|c| if c.get().is_empty() { name.to_string() } else { format!("{}: {name}", c.get()) });
        let e = m.entry(key).or_insert((0.0, 0));
        e.0 += secs;
        e.1 += 1;
    });
    Ok(r)
}

pub fn report() {
    if !enabled() {
        return;
    }
    TIMES.with(|m| {
        let m = m.borrow();
        let mut rows: Vec<_> = m.iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        eprintln!("[profile] section                     total s    calls   ms/call");
        for (name, (secs, n)) in rows {
            eprintln!("[profile] {name:<32} {secs:>9.3} {n:>8} {:>9.3}", 1000.0 * secs / *n as f64);
        }
    });
}
