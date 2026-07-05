use anyhow::{Result, bail};

pub struct BitWriter {
    data: Vec<u8>,
    acc: u128,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    pub fn write(&mut self, value: u64, bitcount: u32) -> Result<()> {
        if bitcount == 0 {
            return Ok(());
        }
        if bitcount > 64 || value as u128 >= (1u128 << bitcount) {
            bail!("value {} does not fit in {} bits", value, bitcount);
        }
        self.acc = (self.acc << bitcount) | value as u128;
        self.nbits += bitcount;
        while self.nbits >= 8 {
            let shift = self.nbits - 8;
            self.data.push(((self.acc >> shift) & 255) as u8);
            self.acc &= (1u128 << shift) - 1;
            self.nbits -= 8;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.data.push(((self.acc << (8 - self.nbits)) & 255) as u8);
            self.acc = 0;
            self.nbits = 0;
        }
        std::mem::take(&mut self.data)
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BitReader<'a> {
    data: &'a [u8],
    i: usize,
    acc: u128,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            i: 0,
            acc: 0,
            nbits: 0,
        }
    }

    pub fn read(&mut self, bitcount: u32) -> Result<u64> {
        if bitcount > 64 {
            bail!("cannot read more than 64 bits at once");
        }
        while self.nbits < bitcount {
            if self.i >= self.data.len() {
                bail!("bitstream ended early");
            }
            self.acc = (self.acc << 8) | self.data[self.i] as u128;
            self.i += 1;
            self.nbits += 8;
        }
        let shift = self.nbits - bitcount;
        let value = (self.acc >> shift) & ((1u128 << bitcount) - 1);
        self.acc &= (1u128 << shift) - 1;
        self.nbits -= bitcount;
        Ok(value as u64)
    }
}

#[derive(Debug, Clone)]
pub struct PBC3Config {
    pub patch_count: u32,
    pub search_depth: u32,
    pub proposal_depth: u32,
    pub exact_depth: u32,
    pub min_patch_size: u32,
    pub max_patch_size: u32,
    pub min_cell_size: u32,
    pub max_cell_size: u32,
    pub cell_sizes_per_candidate: u32,
    pub top_k: u32,
    pub search_q_start: f64,
    pub search_q_end: f64,
    pub q_init: f64,
    pub q_start: f64,
    pub q_end: f64,
    pub color_space: String,
    pub channel_cycle: String,
    pub auto_downsample_init: bool,
    pub init_search_depth: u32,
    pub downsample_init_cell_size: u32,
    pub downsample_palette_bitcount: u32,
    pub downsample_rate: f64,
    pub auto_downsample_max_pixels: u32,
    pub warmup_ratio: f64,
    pub warm_downsample_max_pixels: u32,
    pub patch_palette_bitcount: u32,
    pub quality_target_mae: f64,
    pub mask_size: u32,
    pub anchor_block_size: u32,
    pub positive_bias: bool,
    pub learned_filler_enabled: bool,
    pub learned_filler_model_path: String,
    pub learned_filler_top_k: u32,
    pub learned_filler_q: f64,
    pub learned_filler_candidates: u32,
    pub use_lzma: bool,
    pub random_seed: u32,
    pub compute_final_mse: bool,
    pub debug_mode: bool,
    pub debug_print: bool,
    pub debug_path: Option<String>,
}

fn normalize_channel_cycle(raw: &str) -> String {
    let lower = raw.trim().to_lowercase().replace('_', " ");
    match lower.as_str() {
        "off" | "cycle" | "round robin" | "roundrobin" | "mod" => "Mod".to_string(),
        "sum" | "sum target" | "target" | "max" | "max sum" => "Sum".to_string(),
        _ => raw.to_string(),
    }
}

impl PBC3Config {
    pub fn new() -> Self {
        Self {
            patch_count: 50,
            search_depth: 200,
            proposal_depth: 50,
            exact_depth: 10,
            min_patch_size: 16,
            max_patch_size: 400,
            min_cell_size: 1,
            max_cell_size: 64,
            cell_sizes_per_candidate: 3,
            top_k: 20,
            search_q_start: 0.5,
            search_q_end: 0.2,
            q_init: 0.7,
            q_start: 0.8,
            q_end: 0.8,
            color_space: "YCbCr".to_string(),
            channel_cycle: "Sum".to_string(),
            auto_downsample_init: true,
            init_search_depth: 3,
            downsample_init_cell_size: 12,
            downsample_palette_bitcount: 6,
            downsample_rate: -1.0,
            auto_downsample_max_pixels: 250_000,
            warmup_ratio: -1.0,
            warm_downsample_max_pixels: 750_000,
            patch_palette_bitcount: 2,
            quality_target_mae: 0.0,
            mask_size: 4,
            anchor_block_size: 8,
            positive_bias: true,
            learned_filler_enabled: true,
            learned_filler_model_path: "patch_policy.npz".to_string(),
            learned_filler_top_k: 1,
            learned_filler_q: 0.6,
            learned_filler_candidates: 1,
            use_lzma: true,
            random_seed: 2003,
            compute_final_mse: true,
            debug_mode: false,
            debug_print: false,
            debug_path: None,
        }
    }

    fn apply_cycle_normalization(&mut self) {
        self.channel_cycle = normalize_channel_cycle(&self.channel_cycle);
    }

    fn _preset(mut config: PBC3Config, overrides: Option<PBC3Config>) -> Self {
        if let Some(overrides) = overrides {
            config = Self::merge(config, overrides);
        }
        config.apply_cycle_normalization();
        config
    }

    fn merge(self, overrides: Self) -> Self {
        Self {
            patch_count: overrides.patch_count,
            search_depth: overrides.search_depth,
            proposal_depth: overrides.proposal_depth,
            exact_depth: overrides.exact_depth,
            min_patch_size: overrides.min_patch_size,
            max_patch_size: overrides.max_patch_size,
            min_cell_size: overrides.min_cell_size,
            max_cell_size: overrides.max_cell_size,
            cell_sizes_per_candidate: overrides.cell_sizes_per_candidate,
            top_k: overrides.top_k,
            search_q_start: overrides.search_q_start,
            search_q_end: overrides.search_q_end,
            q_init: overrides.q_init,
            q_start: overrides.q_start,
            q_end: overrides.q_end,
            color_space: overrides.color_space,
            channel_cycle: overrides.channel_cycle,
            auto_downsample_init: overrides.auto_downsample_init,
            init_search_depth: overrides.init_search_depth,
            downsample_init_cell_size: overrides.downsample_init_cell_size,
            downsample_palette_bitcount: overrides.downsample_palette_bitcount,
            downsample_rate: overrides.downsample_rate,
            auto_downsample_max_pixels: overrides.auto_downsample_max_pixels,
            warmup_ratio: overrides.warmup_ratio,
            warm_downsample_max_pixels: overrides.warm_downsample_max_pixels,
            patch_palette_bitcount: overrides.patch_palette_bitcount,
            quality_target_mae: overrides.quality_target_mae,
            mask_size: overrides.mask_size,
            anchor_block_size: overrides.anchor_block_size,
            positive_bias: overrides.positive_bias,
            learned_filler_enabled: overrides.learned_filler_enabled,
            learned_filler_model_path: overrides.learned_filler_model_path,
            learned_filler_top_k: overrides.learned_filler_top_k,
            learned_filler_q: overrides.learned_filler_q,
            learned_filler_candidates: overrides.learned_filler_candidates,
            use_lzma: overrides.use_lzma,
            random_seed: overrides.random_seed,
            compute_final_mse: overrides.compute_final_mse,
            debug_mode: overrides.debug_mode,
            debug_print: overrides.debug_print,
            debug_path: overrides.debug_path,
            ..self
        }
    }

    pub fn compression(overrides: Option<PBC3Config>) -> Self {
        Self::_preset(
            Self {
                patch_count: 50,
                search_q_start: 0.5,
                search_q_end: 0.2,
                init_search_depth: 3,
                q_init: 0.7,
                q_start: 0.8,
                q_end: 0.8,
                quality_target_mae: 0.0,
                learned_filler_enabled: true,
                learned_filler_q: 0.4,
                ..Self::new()
            },
            overrides,
        )
    }

    pub fn balanced(overrides: Option<PBC3Config>) -> Self {
        Self::_preset(
            Self {
                patch_count: 50,
                search_q_start: 0.5,
                search_q_end: 0.2,
                init_search_depth: 3,
                q_init: 0.7,
                q_start: 0.8,
                q_end: 0.8,
                quality_target_mae: 0.0,
                learned_filler_enabled: true,
                learned_filler_q: 0.6,
                ..Self::new()
            },
            overrides,
        )
    }

    pub fn quality(overrides: Option<PBC3Config>) -> Self {
        Self::_preset(
            Self {
                patch_count: 50,
                search_q_start: 0.5,
                search_q_end: 0.2,
                init_search_depth: 3,
                q_init: 0.7,
                q_start: 0.8,
                q_end: 0.8,
                quality_target_mae: 0.0,
                learned_filler_enabled: true,
                learned_filler_q: 0.8,
                ..Self::new()
            },
            overrides,
        )
    }

    pub fn high_quality(overrides: Option<PBC3Config>) -> Self {
        Self::_preset(
            Self {
                patch_count: 20,
                search_q_start: 0.7,
                search_q_end: 0.2,
                init_search_depth: 3,
                q_init: 0.7,
                q_start: 0.8,
                q_end: 0.8,
                quality_target_mae: 0.0,
                learned_filler_enabled: true,
                learned_filler_q: 0.95,
                ..Self::new()
            },
            overrides,
        )
    }
}

impl Default for PBC3Config {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PBC3Result {
    pub image: Vec<u8>,
    pub image_width: u32,
    pub image_height: u32,
    pub data: Vec<u8>,
    pub config: PBC3Config,
    pub mse: Option<f64>,
    pub encode_seconds: f64,
    pub total_bits: u64,
    pub original_width: Option<u32>,
    pub original_height: Option<u32>,
    pub working_width: Option<u32>,
    pub working_height: Option<u32>,
    pub debug_path: Option<String>,
    pub channels: u32,
}

impl PBC3Result {
    pub fn time(&self) -> f64 {
        self.encode_seconds
    }

    pub fn encode_time(&self) -> f64 {
        self.encode_seconds
    }

    pub fn decode_time(&self) -> f64 {
        self.encode_seconds
    }

    pub fn decode_seconds(&self) -> f64 {
        self.encode_seconds
    }

    pub fn original_bits(&self) -> u64 {
        let w = self.original_width.unwrap_or(self.image_width) as u64;
        let h = self.original_height.unwrap_or(self.image_height) as u64;
        w * h * self.channels as u64 * 8
    }

    pub fn compressed_kb(&self) -> f64 {
        self.total_bits as f64 / 8.0 / 1024.0
    }

    pub fn original_kb(&self) -> f64 {
        self.original_bits() as f64 / 8.0 / 1024.0
    }

    pub fn compression_rate(&self) -> f64 {
        if self.total_bits == 0 {
            f64::INFINITY
        } else {
            self.original_bits() as f64 / self.total_bits as f64
        }
    }

    pub fn compressed_percent(&self) -> f64 {
        let ob = self.original_bits();
        if ob == 0 {
            0.0
        } else {
            self.total_bits as f64 / ob as f64 * 100.0
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        std::fs::write(path, &self.data)?;
        Ok(())
    }

    pub fn verify(&self) -> bool {
        // The Rust decoder is not ported yet; keep the method present but conservative.
        false
    }
}
