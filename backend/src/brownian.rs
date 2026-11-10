//! The Brownian noise of ComfyUI's SDE samplers (k_diffusion's BrownianTreeNoiseSampler): a
//! torchsde 0.2.6 BrownianTree (a BrownianInterval with `halfway_tree`, tol 1e-6, pool_size 24,
//! no Levy area) whose every node draws its noise from `torch.randn` with a seed from numpy's
//! SeedSequence, on the CPU (`cpu=True`, dpmpp_2m_sde) or on the card (dpmpp_2m_sde_gpu, whose
//! draws follow the card's SM count as all of torch's CUDA randn does). Each value is a function of the seed and the queried
//! times only (the tree splits at rounded halfway points; the LRU cache torchsde keeps changes
//! nothing but speed), so the same queries give ComfyUI's values.

use crate::rng::{torch_randn_cpu, CudaRandn};

/// Where the tree's nodes draw their noise
pub enum NoiseSource {
    Cpu,
    Cuda { sms: u32, dev: candle_core::Device },
}

impl NoiseSource {
    /// torch.randn(size, generator=torch.Generator(device).manual_seed(seed)), float32
    fn randn(&self, seed: u32, size: usize) -> anyhow::Result<Vec<f32>> {
        Ok(match self {
            NoiseSource::Cpu => torch_randn_cpu(seed as u64, size),
            NoiseSource::Cuda { sms, dev } => CudaRandn::new(seed as u64, *sms, 1536).randn_gpu(size, dev)?.to_vec1()?,
        })
    }
}

// numpy/random/bit_generator.pyx
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> 16)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> 16)
}

/// _int_to_uint32_array: a non-negative int as little-endian 32-bit words ([0] for 0)
fn int_words(mut n: u64) -> Vec<u32> {
    if n == 0 {
        return vec![0];
    }
    let mut out = vec![];
    while n > 0 {
        out.push(n as u32);
        n >>= 32;
    }
    out
}

/// `np.random.SeedSequence(entropy, spawn_key=spawn_key, pool_size=pool_size).generate_state(n)`
pub fn seed_sequence_state(entropy: u64, spawn_key: &[u64], pool_size: usize, n: usize) -> Vec<u32> {
    // get_assembled_entropy: the run entropy padded with zeros to the pool size when a spawn key
    // follows it (numpy 1.19's fix), then the spawn key's words
    let mut words = int_words(entropy);
    let spawn: Vec<u32> = spawn_key.iter().flat_map(|&k| int_words(k)).collect();
    if !spawn.is_empty() && words.len() < pool_size {
        words.resize(pool_size, 0);
    }
    words.extend(spawn);
    // mix_entropy
    let mut pool = vec![0u32; pool_size];
    let mut hc = INIT_A;
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(words.get(i).copied().unwrap_or(0), &mut hc);
    }
    for i_src in 0..pool_size {
        for i_dst in 0..pool_size {
            if i_src != i_dst {
                let h = hashmix(pool[i_src], &mut hc);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }
    for &w in words.iter().skip(pool_size) {
        for i_dst in 0..pool_size {
            let h = hashmix(w, &mut hc);
            pool[i_dst] = mix(pool[i_dst], h);
        }
    }
    // generate_state
    let mut hc = INIT_B;
    (0..n)
        .map(|i| {
            let mut d = pool[i % pool_size] ^ hc;
            hc = hc.wrapping_mul(MULT_B);
            d = d.wrapping_mul(hc);
            d ^ (d >> 16)
        })
        .collect()
}

/// Python's round(x, 6) on a float: the correctly rounded 6-decimal value, read back
fn round6(x: f64) -> f64 {
    format!("{x:.6}").parse().expect("a formatted float parses")
}

struct Node {
    start: f64,
    end: f64,
    parent: Option<usize>,
    is_left: bool,
    midway: Option<f64>,
    spawn_key: u64,
    depth: u64,
    w_seed: u32,
    children: (usize, usize),
}

/// torchsde's BrownianTree(t0, zeros, t1, entropy=seed) over tensors of `size` f32 values
pub struct BrownianTree {
    entropy: u64,
    size: usize,
    source: NoiseSource,
    nodes: Vec<Node>,
    w: std::collections::HashMap<usize, Vec<f32>>,
}

const POOL: usize = 24;

impl BrownianTree {
    pub fn new(t0: f64, t1: f64, entropy: u64, size: usize, source: NoiseSource) -> anyhow::Result<Self> {
        // BrownianInterval.__init__: W = randn(initial_W_seed) * sqrt(t1 - t0), the times
        // unrounded there; the top interval's own ends are rounded
        let top = seed_sequence_state(entropy, &[], POOL, 3);
        let scale = (t1 - t0).sqrt() as f32;
        let w_top: Vec<f32> = source.randn(top[0], size)?.into_iter().map(|v| v * scale).collect();
        let root = Node {
            start: round6(t0),
            end: round6(t1),
            parent: None,
            is_left: false,
            midway: None,
            spawn_key: 0,
            depth: 0,
            w_seed: 0,
            children: (0, 0),
        };
        let mut w = std::collections::HashMap::new();
        w.insert(0, w_top);
        Ok(Self { entropy, size, source, nodes: vec![root], w })
    }

    /// _split_exact: the node splits at `m`, its seeds from SeedSequence(entropy, spawn_key=(key,
    /// depth)), the children made with the unrounded point (their ends round it)
    fn split_exact(&mut self, i: usize, m: f64) {
        let (key, depth) = match self.nodes[i].parent {
            None => (0, 0),
            Some(p) => (2 * self.nodes[p].spawn_key + if self.nodes[i].is_left { 0 } else { 1 }, self.nodes[p].depth + 1),
        };
        let seeds = seed_sequence_state(self.entropy, &[key, depth], POOL, 4);
        let (start, end) = (self.nodes[i].start, self.nodes[i].end);
        let mid = round6(m);
        let l = self.nodes.len();
        self.nodes.push(Node {
            start: round6(start),
            end: round6(m),
            parent: Some(i),
            is_left: true,
            midway: None,
            spawn_key: 0,
            depth: 0,
            w_seed: 0,
            children: (0, 0),
        });
        self.nodes.push(Node {
            start: round6(m),
            end: round6(end),
            parent: Some(i),
            is_left: false,
            midway: None,
            spawn_key: 0,
            depth: 0,
            w_seed: 0,
            children: (0, 0),
        });
        let n = &mut self.nodes[i];
        n.midway = Some(mid);
        n.spawn_key = key;
        n.depth = depth;
        n.w_seed = seeds[0];
        n.children = (l, l + 1);
    }

    /// _split with halfway_tree: at the rounded halfway point, then on down towards `m`
    fn split(&mut self, i: usize, m: f64) {
        let half = 0.5 * (self.nodes[i].end + self.nodes[i].start);
        self.split_exact(i, half);
        let mid = self.nodes[i].midway.unwrap();
        let (l, r) = self.nodes[i].children;
        if m > mid {
            self.split(r, m);
        } else if m < mid {
            self.split(l, m);
        }
    }

    /// _loc_inner: the existing (or newly split) intervals that make up [ta, tb]. A query outside
    /// this interval goes to its parent, which is where a split leaves it when the halfway point
    /// falls before tb.
    fn loc(&mut self, i: usize, ta: f64, tb: f64, out: &mut Vec<usize>) {
        let (start, end) = (self.nodes[i].start, self.nodes[i].end);
        if ta < start || tb > end {
            let p = self.nodes[i].parent.expect("the top interval covers every query");
            return self.loc(p, ta, tb, out);
        }
        if ta == start && tb == end {
            out.push(i);
            return;
        }
        let Some(mid) = self.nodes[i].midway else {
            if ta == start {
                self.split(i, tb);
                let l = self.nodes[i].children.0;
                return self.loc(l, ta, tb, out);
            }
            self.split(i, ta);
            let r = self.nodes[i].children.1;
            return self.loc(r, ta, tb, out);
        };
        let (l, r) = self.nodes[i].children;
        if tb <= mid {
            self.loc(l, ta, tb, out)
        } else if ta >= mid {
            self.loc(r, ta, tb, out)
        } else {
            self.loc(l, ta, mid, out);
            self.loc(r, mid, tb, out);
        }
    }

    /// An interval's increment, from its parent's (_increment_and_space_time_levy_area without
    /// H): the bridge mean left_diff * W * (1 / h) plus sqrt(left_diff * right_diff / h) times the
    /// parent's noise, each scalar rounded to f32 as torch multiplies a float tensor by a python float
    fn increment(&mut self, i: usize) -> anyhow::Result<Vec<f32>> {
        if let Some(w) = self.w.get(&i) {
            return Ok(w.clone());
        }
        let p = self.nodes[i].parent.expect("only the top has no parent, and its W is known");
        let wp = self.increment(p)?;
        let (ps, pe, pm) = (self.nodes[p].start, self.nodes[p].end, self.nodes[p].midway.unwrap());
        let h_reciprocal = 1.0 / (pe - ps);
        let left_diff = pm - ps;
        let right_diff = pe - pm;
        let var = left_diff * right_diff * h_reciprocal;
        let noise = self.source.randn(self.nodes[p].w_seed, self.size)?;
        let (ld, hr, sd) = (left_diff as f32, h_reciprocal as f32, var.sqrt() as f32);
        let left_w: Vec<f32> = wp.iter().zip(&noise).map(|(&w, &n)| (ld * w) * hr + sd * n).collect();
        let out = if self.nodes[i].is_left { left_w } else { wp.iter().zip(&left_w).map(|(&w, &l)| w - l).collect() };
        self.w.insert(i, out.clone());
        Ok(out)
    }

    /// BrownianInterval(ta, tb): W(tb) - W(ta) for t0 <= ta < tb <= t1 (times as the f32 values
    /// the sampler hands over, read as doubles), the pieces' increments summed in order
    pub fn query(&mut self, ta: f64, tb: f64) -> anyhow::Result<Vec<f32>> {
        if ta == tb {
            return Ok(vec![0.0; self.size]);
        }
        let (ta, tb) = (round6(ta), round6(tb));
        let mut parts = vec![];
        self.loc(0, ta, tb, &mut parts);
        let mut w = self.increment(parts[0])?;
        for &q in &parts[1..] {
            let wi = self.increment(q)?;
            for (a, b) in w.iter_mut().zip(&wi) {
                *a += b;
            }
        }
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    // numpy 2.5.2's SeedSequence(...).generate_state, from the reference ComfyUI's venv
    #[test]
    fn seed_sequence_like_numpy() {
        assert_eq!(super::seed_sequence_state(4242, &[], 24, 3), [0x6f24bb5a, 0x6cf6654c, 0xb5eebef5]);
        assert_eq!(super::seed_sequence_state(4242, &[1, 1], 24, 4), [0x2d01ff16, 0xa6172424, 0x09086563, 0x29918904]);
        assert_eq!(super::seed_sequence_state((1 << 40) + 7, &[5, 3], 24, 4), [0x10617802, 0x76f29aea, 0xb7bdd303, 0xaa646b95]);
    }

    // torchsde 0.2.6's BrownianTree(t0, zeros(16), t1, entropy=20261014, tol=1e-6, pool_size=24)
    // queried over three karras steps, the times float32 as the sampler hands them over: the
    // first four values and the sum of all sixteen values' bits
    #[test]
    fn brownian_tree_like_torchsde() {
        let f = |v: f32| v as f64;
        let mut tree = super::BrownianTree::new(f(0.0291675), f(14.614642), 20261014, 16, super::NoiseSource::Cpu).unwrap();
        for ((a, b), first, sum) in [
            ((12.660604f32, 14.614642f32), [0xbfab8f77u32, 0xbe9b6dc8, 0x3e1e6a92, 0xbfc04eb3], 0xf6c49be7u32),
            ((10.934923, 12.660604), [0xbe1097b0, 0xbe8719a0, 0x3fb83be4, 0xbfc75762], 0xf193bc86),
            ((0.0291675, 0.040858123), [0xbdc78421, 0xbc9932e9, 0xbe152e41, 0xbdd90086], 0x55ae20d6),
        ] {
            let w = tree.query(f(a), f(b)).unwrap();
            let bits: Vec<u32> = w.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits[..4], first);
            assert_eq!(bits.iter().fold(0u32, |s, b| s.wrapping_add(*b)), sum);
        }
    }

    #[test]
    fn rounds_like_python() {
        assert_eq!(super::round6(0.0291675329208374), 0.029168);
        assert_eq!(super::round6(14.614641189575195), 14.614641);
        assert_eq!(super::round6(0.5 * (14.614641 + 0.029168)), 7.321905);
    }
}
