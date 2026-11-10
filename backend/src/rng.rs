//! PyTorch's random numbers, reproduced, so a seed means the same thing here as in ComfyUI.
//!
//! ComfyUI draws two kinds of noise:
//!   * the starting latent: `torch.randn(..., generator=torch.manual_seed(seed), device="cpu")`,
//!     i.e. MT19937 plus PyTorch's Box-Muller over blocks of 16 (`normal_fill`);
//!   * each step of an ancestral sampler: `torch.randn(..., device="cuda", generator=g)` with
//!     `g.manual_seed(seed)`, i.e. Philox4x32-10 + curand's Box-Muller, laid out by PyTorch's
//!     grid-stride kernel. That layout depends on the card's SM count, so the same seed gives
//!     different step noise on an RTX 4070 Ti SUPER and an RTX 4090, in ComfyUI and here alike.

use std::f64::consts::PI;

// ── MT19937, as at::mt19937 ──────────────────────────────────────────────────
pub struct Mt19937 {
    state: [u32; 624],
    next: usize,
}

impl Mt19937 {
    /// `torch.manual_seed(seed)`: only the low 32 bits seed the engine.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..624 {
            state[j] = 1812433253u32.wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30)).wrapping_add(j as u32);
        }
        Mt19937 { state, next: 624 }
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        for i in 0..624 {
            let y = (self.state[i] & UPPER) | (self.state[(i + 1) % 624] & LOWER);
            let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.state[i] = v;
        }
        self.next = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.next >= 624 {
            self.twist();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// at::uniform_real_distribution<float>(0, 1): the low 24 bits over 2^24.
    fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00ff_ffff) as f32 * (1.0f32 / 16_777_216.0)
    }

    /// Python's `random.Random(seed)` for a whole number: init_by_array over the number's 32-bit
    /// words, low word first (one zero word for 0)
    pub fn python(seed: u64) -> Self {
        let key: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        let mut m = Mt19937::new(19650218);
        let mt = &mut m.state;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        m
    }

    /// Python's random.shuffle: from the end, each place swapped with one below it or itself,
    /// drawn by _randbelow (getrandbits of n's bit length, again until under n)
    pub fn python_shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let n = i as u64 + 1;
            let k = 64 - n.leading_zeros();
            let j = loop {
                let r = (self.next_u32() >> (32 - k)) as u64;
                if r < n {
                    break r as usize;
                }
            };
            x.swap(i, j);
        }
    }
}

#[cfg(test)]
mod py_tests {
    use super::Mt19937;

    #[test]
    fn python_random_like_cpython() {
        // values from CPython 3.12 (ComfyUI's): random.Random(seed).getrandbits(32), .shuffle
        let mut r = Mt19937::python(2613179247);
        assert_eq!([r.next_u32(), r.next_u32(), r.next_u32()], [3111975202, 1266694857, 3062096946]);
        let mut r = Mt19937::python(0);
        assert_eq!([r.next_u32(), r.next_u32(), r.next_u32()], [3626764237, 1654615998, 3255389356]);
        for (seed, n, want) in [
            (0u64, 3usize, vec![0usize, 2, 1]),
            (0, 5, vec![2, 1, 0, 4, 3]),
            (1, 5, vec![2, 3, 4, 0, 1]),
            (2613179247, 5, vec![1, 3, 0, 4, 2]),
            (18446744073709551615, 5, vec![2, 4, 3, 1, 0]),
            (123456789012345, 3, vec![1, 0, 2]),
        ] {
            let mut x: Vec<usize> = (0..n).collect();
            Mt19937::python(seed).python_shuffle(&mut x);
            assert_eq!(x, want, "seed {seed}, {n} items");
        }
    }
}

// torch 2.11's CPU randn on an AVX2 or AVX-512 machine (both boxes, and ComfyUI's container)
// is normal_fill_16_AVX2: avx_mathfun's Cephes log256_ps and sincos256_ps, theta = (float)2pi * u2
// in float. Its build (-mavx2 -mfma) lets GCC fuse each multiply into the add that consumes it,
// so FMA_FUSED picks between the fused and unfused forms; the scalar libm path below is what
// torch takes with ATEN_CPU_CAPABILITY=default only.

const FMA_FUSED: bool = true;

#[inline]
fn mad(a: f32, b: f32, c: f32) -> f32 {
    if FMA_FUSED {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// avx_mathfun log256_ps, one lane
fn log256(x: f32) -> f32 {
    const SQRTHF: f32 = 0.707106781186547524;
    const P: [f32; 9] = [
        7.0376836292E-2,
        -1.1514610310E-1,
        1.1676998740E-1,
        -1.2420140846E-1,
        1.4249322787E-1,
        -1.6668057665E-1,
        2.0000714765E-1,
        -2.4999993993E-1,
        3.3333331174E-1,
    ];
    const Q1: f32 = -2.12194440e-4;
    const Q2: f32 = 0.693359375;
    let invalid = x <= 0.0;
    let x = if x > f32::from_bits(0x0080_0000) { x } else { f32::from_bits(0x0080_0000) };
    let imm0 = (x.to_bits() >> 23) as i32 - 0x7f;
    let mut xm = f32::from_bits((x.to_bits() & !0x7f80_0000u32) | 0.5f32.to_bits());
    let mut e = imm0 as f32 + 1.0;
    let mask = xm < SQRTHF;
    let tmp = if mask { xm } else { 0.0 };
    xm -= 1.0;
    e -= if mask { 1.0 } else { 0.0 };
    xm += tmp;
    let z = xm * xm;
    let mut y = P[0];
    for p in &P[1..] {
        y = mad(y, xm, *p);
    }
    // y*xm stays a product; GCC fuses the later multiply, y*z, into the add with e*Q1
    let ya = y * xm;
    y = mad(ya, z, e * Q1);
    y = mad(-z, 0.5, y);
    let x = mad(e, Q2, xm + y);
    if invalid {
        f32::NAN
    } else {
        x
    }
}

/// avx_mathfun sincos256_ps, one lane: (sin, cos)
fn sincos256(x: f32) -> (f32, f32) {
    const FOPI: f32 = 1.27323954473516;
    const DP: [f32; 3] = [-0.78515625, -2.4187564849853515625e-4, -3.77489497744594108e-8];
    const SIN: [f32; 3] = [-1.9515295891E-4, 8.3321608736E-3, -1.6666654611E-1];
    const COS: [f32; 3] = [2.443315711809948E-005, -1.388731625493765E-003, 4.166664568298827E-002];
    let sign_in = x.to_bits() & 0x8000_0000;
    let mut xa = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    let yv = xa * FOPI;
    let mut imm2 = yv as i32;
    imm2 = imm2.wrapping_add(1) & !1;
    let yf = imm2 as f32;
    let swap_sin = ((imm2 & 4) as u32) << 29;
    let poly_mask = (imm2 & 2) == 0;
    xa = mad(yf, DP[0], xa);
    xa = mad(yf, DP[1], xa);
    xa = mad(yf, DP[2], xa);
    let sign_cos = (((!imm2.wrapping_sub(2)) & 4) as u32) << 29;
    let sign_sin = sign_in ^ swap_sin;
    let z = xa * xa;
    let mut y = mad(COS[0], z, COS[1]);
    y = mad(y, z, COS[2]);
    // as in log256: the second y*z is fused into the subtraction of z*0.5
    let ya = y * z;
    y = mad(ya, z, -(z * 0.5));
    y += 1.0;
    let mut y2 = mad(SIN[0], z, SIN[1]);
    y2 = mad(y2, z, SIN[2]);
    y2 *= z;
    y2 = mad(y2, xa, xa);
    let ysin2 = if poly_mask { y2 } else { 0.0 };
    let ysin1 = if poly_mask { 0.0 } else { y };
    let y2r = y2 - ysin2;
    let yr = y - ysin1;
    let s = ysin1 + ysin2;
    let c = yr + y2r;
    (f32::from_bits(s.to_bits() ^ sign_sin), f32::from_bits(c.to_bits() ^ sign_cos))
}

/// torch's normal_fill_16_AVX2 (mean 0, std 1: the final fmadd leaves the product as it is)
fn normal_fill_16_avx2(data: &mut [f32]) {
    let two_pi = (2.0f64 * PI) as f32;
    for j in 0..8 {
        let u1 = 1.0f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f32 * log256(u1)).sqrt();
        let theta = two_pi * u2;
        let (s, c) = sincos256(theta);
        data[j] = radius * c;
        data[j + 8] = radius * s;
    }
}

/// at::native normal_fill_16, the scalar path (ATEN_CPU_CAPABILITY=default): u1 = 1 - a,
/// theta computed in double then narrowed, libm's logf, cosf and sinf.
#[allow(dead_code)]
fn normal_fill_16(data: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = (2.0f64 * PI * u2 as f64) as f32;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

/// `torch.randn(n, generator=torch.manual_seed(seed))` on the CPU, float32.
pub fn torch_randn_cpu(seed: u64, n: usize) -> Vec<f32> {
    TorchCpuGenerator::new(seed).randn(n)
}

/// torch's CPU generator after `torch.manual_seed(seed)`: successive `torch.randn` calls draw
/// on from where the last one stopped (ComfyUI's prepare_noise does this for a nested latent).
pub struct TorchCpuGenerator {
    mt: Mt19937,
}

impl TorchCpuGenerator {
    pub fn new(seed: u64) -> Self {
        Self { mt: Mt19937::new(seed) }
    }

    /// `torch.randn(n, generator=g)`, float32
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        let g = &mut self.mt;
        let mut data: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
        if n < 16 {
            // PyTorch's small-tensor path draws pairs one value at a time; nothing here needs it
            panic!("torch randn: fewer than 16 values is not implemented");
        }
        let mut i = 0;
        while i + 16 <= n {
            normal_fill_16_avx2(&mut data[i..i + 16]);
            i += 16;
        }
        if !n.is_multiple_of(16) {
            // PyTorch recomputes the last 16 values from fresh uniforms
            let tail = n - 16;
            for v in data[tail..].iter_mut() {
                *v = g.uniform_f32();
            }
            normal_fill_16_avx2(&mut data[tail..]);
        }
        data
    }
}

// ── Philox4x32-10, as curand ────────────────────────────────────────────────
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;

fn philox_round(c: [u32; 4], k: [u32; 2]) -> [u32; 4] {
    let p0 = (M0 as u64) * (c[0] as u64);
    let p1 = (M1 as u64) * (c[2] as u64);
    let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
    let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
    [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0]
}

fn philox(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        if round > 0 {
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        c = philox_round(c, k);
    }
    c
}

struct PhiloxState {
    ctr: [u32; 4],
    key: [u32; 2],
    output: [u32; 4],
    state: u32,
}

impl PhiloxState {
    /// curand_init(seed, subsequence, offset)
    fn new(seed: u64, subsequence: u64, offset: u64) -> Self {
        let mut s = PhiloxState { ctr: [0; 4], key: [seed as u32, (seed >> 32) as u32], output: [0; 4], state: 0 };
        // skipahead_sequence: add to the high 64 bits of the counter
        let (nlo, mut nhi) = (subsequence as u32, (subsequence >> 32) as u32);
        s.ctr[2] = s.ctr[2].wrapping_add(nlo);
        if s.ctr[2] < nlo {
            nhi = nhi.wrapping_add(1);
        }
        s.ctr[3] = s.ctr[3].wrapping_add(nhi);
        s.output = philox(s.ctr, s.key);
        // skipahead(offset)
        let mut n = offset;
        s.state += (n & 3) as u32;
        n /= 4;
        if s.state > 3 {
            n += 1;
            s.state -= 4;
        }
        s.incr_by(n);
        s.output = philox(s.ctr, s.key);
        s
    }

    fn incr_by(&mut self, n: u64) {
        let (nlo, mut nhi) = (n as u32, (n >> 32) as u32);
        self.ctr[0] = self.ctr[0].wrapping_add(nlo);
        if self.ctr[0] < nlo {
            nhi = nhi.wrapping_add(1);
        }
        self.ctr[1] = self.ctr[1].wrapping_add(nhi);
        if nhi <= self.ctr[1] {
            return;
        }
        self.ctr[2] = self.ctr[2].wrapping_add(1);
        if self.ctr[2] != 0 {
            return;
        }
        self.ctr[3] = self.ctr[3].wrapping_add(1);
    }

    fn incr(&mut self) {
        for i in 0..4 {
            self.ctr[i] = self.ctr[i].wrapping_add(1);
            if self.ctr[i] != 0 {
                return;
            }
        }
    }

    /// curand4
    fn next4(&mut self) -> [u32; 4] {
        let tmp = self.output;
        self.incr();
        self.output = philox(self.ctr, self.key);
        let o = self.output;
        match self.state {
            0 => tmp,
            1 => [tmp[1], tmp[2], tmp[3], o[0]],
            2 => [tmp[2], tmp[3], o[0], o[1]],
            _ => [tmp[3], o[0], o[1], o[2]],
        }
    }
}

const TWO_POW32_INV: f32 = 2.3283064e-10;
const TWO_POW32_INV_2PI: f32 = 2.3283064e-10 * 6.2831855;

/// curand's _curand_box_muller. nvcc contracts `x * a + b` into an FMA; the device's
/// __sincosf is a hardware approximation, so sin/cos here can differ in the last bits.
fn box_muller(x: u32, y: u32) -> (f32, f32) {
    let u = (x as f32).mul_add(TWO_POW32_INV, TWO_POW32_INV / 2.0);
    let v = (y as f32).mul_add(TWO_POW32_INV_2PI, TWO_POW32_INV_2PI / 2.0);
    let s = (-2.0f32 * u.ln()).sqrt();
    (v.sin() * s, v.cos() * s)
}

/// A `torch.Generator(device="cuda")` after `manual_seed(seed)`, drawing float32 randn the way
/// PyTorch's distribution_nullary_kernel lays the draws out (block 256, unroll 4, grid capped
/// at SMs * (max threads per SM / 256)).
pub struct CudaRandn {
    seed: u64,
    offset: u64,
    sms: u64,
    blocks_per_sm: u64,
}

impl CudaRandn {
    pub fn new(seed: u64, sms: u32, max_threads_per_sm: u32) -> Self {
        CudaRandn { seed, offset: 0, sms: sms as u64, blocks_per_sm: (max_threads_per_sm / 256) as u64 }
    }

    /// The next `numel` values on the GPU, made by curand's own device code as torch makes them:
    /// bit for bit torch's (`randn` below is the CPU port, about 1e-7 off)
    pub fn randn_gpu(&mut self, numel: usize, dev: &candle_core::Device) -> candle_core::Result<candle_core::Tensor> {
        let n = numel as u64;
        const BLOCK: u64 = 256;
        const UNROLL: u64 = 4;
        let grid = n.div_ceil(BLOCK).min(self.sms * self.blocks_per_sm);
        let threads = BLOCK * grid;
        let offset = self.offset;
        self.offset += ((n - 1) / (threads * UNROLL) + 1) * 4;
        crate::kernels::torch_randn(numel, self.seed, offset, grid as u32, dev)
    }

    /// torch.randint(0, 256, numel, dtype=uint8) from this generator, on the GPU (the same layout and
    /// offset step as randn: curand4, unrolled by 4)
    pub fn randint_u8_gpu(&mut self, numel: usize, dev: &candle_core::Device) -> candle_core::Result<candle_core::Tensor> {
        let n = numel as u64;
        const BLOCK: u64 = 256;
        const UNROLL: u64 = 4;
        let grid = n.div_ceil(BLOCK).min(self.sms * self.blocks_per_sm);
        let threads = BLOCK * grid;
        let offset = self.offset;
        self.offset += ((n - 1) / (threads * UNROLL) + 1) * 4;
        crate::kernels::torch_randint_u8(numel, self.seed, offset, grid as u32, dev)
    }

    pub fn randn(&mut self, numel: usize) -> Vec<f32> {
        let n = numel as u64;
        const BLOCK: u64 = 256;
        const UNROLL: u64 = 4;
        let grid = n.div_ceil(BLOCK).min(self.sms * self.blocks_per_sm);
        let threads = BLOCK * grid;
        let counter_offset = ((n - 1) / (threads * UNROLL) + 1) * 4;
        let offset = self.offset;
        self.offset += counter_offset;

        let mut out = vec![0f32; numel];
        let rounded = ((n - 1) / (threads * UNROLL) + 1) * threads * UNROLL;
        for idx in 0..threads {
            let mut st = PhiloxState::new(self.seed, idx, offset);
            let mut linear = idx;
            while linear < rounded {
                let r = st.next4();
                let (a, b) = box_muller(r[0], r[1]);
                let (c, d) = box_muller(r[2], r[3]);
                for (ii, val) in [a, b, c, d].into_iter().enumerate() {
                    let li = linear + threads * ii as u64;
                    if li < n {
                        out[li as usize] = val;
                    }
                }
                linear += threads * UNROLL;
            }
        }
        out
    }
}
