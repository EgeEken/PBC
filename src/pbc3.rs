use std::cmp::min;
use std::ffi::c_void;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use image::{ImageBuffer, Rgba, imageops::FilterType};
use lzma_sys as lzma;

use crate::palette::{ceil_div, palette_generator, resolve_palette_bitcount};
use crate::types::BitReader;

const MAGIC: [u8; 4] = *b"PBC3";
const VERSION: u8 = 0;
const ENTROPY_STORE: u8 = 0;
const ENTROPY_LZMA: u8 = 2;
const PALETTE_GENERATED: u8 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorSpace {
    Rgb,
    YCbCr,
}

impl ColorSpace {
    fn from_id(id: u64) -> Result<Self> {
        match id {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::YCbCr),
            _ => bail!("unsupported color space id {}", id),
        }
    }
}

#[derive(Debug, Clone)]
struct Warmup {
    w: usize,
    h: usize,
    split: usize,
}

#[derive(Debug, Clone)]
struct Header {
    downsampled: bool,
    original_w: Option<usize>,
    original_h: Option<usize>,
    w: usize,
    h: usize,
    color_space: ColorSpace,
    channels: usize,
    channel_bits: u32,
    positive_bias: bool,
    has_alpha: bool,
    patch_count: usize,
    base_values: Vec<u8>,
    warmup: Option<Warmup>,
}

#[derive(Debug, Clone)]
struct Patch {
    channel: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    cell_size: usize,
    values: Vec<i16>,
}

#[derive(Debug, Clone)]
struct Canvas {
    w: usize,
    h: usize,
    channels: usize,
    data: Vec<i32>,
}

impl Canvas {
    fn new(w: usize, h: usize, channels: usize, base_values: &[u8]) -> Self {
        let mut data = vec![0i32; w * h * channels];
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    data[((y * w + x) * channels) + c] = base_values[c] as i32;
                }
            }
        }
        Self {
            w,
            h,
            channels,
            data,
        }
    }

    fn resize(&self, new_w: usize, new_h: usize) -> Self {
        if (self.w, self.h) == (new_w, new_h) {
            return self.clone();
        }
        let mut out = vec![0i32; new_w * new_h * self.channels];
        for c in 0..self.channels {
            let mut src = vec![0f32; self.w * self.h];
            for y in 0..self.h {
                for x in 0..self.w {
                    src[y * self.w + x] = self.get(x, y, c) as f32;
                }
            }
            let resized = resize_f32(&src, self.w, self.h, new_w, new_h);
            for y in 0..new_h {
                for x in 0..new_w {
                    out[((y * new_w + x) * self.channels) + c] =
                        resized[y * new_w + x].round() as i32;
                }
            }
        }
        Self {
            w: new_w,
            h: new_h,
            channels: self.channels,
            data: out,
        }
    }

    fn get(&self, x: usize, y: usize, c: usize) -> i32 {
        self.data[((y * self.w + x) * self.channels) + c]
    }

    fn add_patch(&mut self, patch: &Patch) {
        let gw = ceil_div(patch.w, patch.cell_size);
        let gh = ceil_div(patch.h, patch.cell_size);
        let resized = resize_i16_to_f32(&patch.values, gw, gh, patch.w, patch.h);
        for py in 0..patch.h {
            for px in 0..patch.w {
                let dst =
                    (((patch.y + py) * self.w + (patch.x + px)) * self.channels) + patch.channel;
                self.data[dst] += resized[py * patch.w + px].round() as i32;
            }
        }
    }

    fn to_rgba_bytes(&self, color_space: ColorSpace, has_alpha: bool) -> Vec<u8> {
        let mut out = vec![0u8; self.w * self.h * 4];
        for y in 0..self.h {
            for x in 0..self.w {
                let i = (y * self.w + x) * self.channels;
                let (r, g, b) = match color_space {
                    ColorSpace::Rgb => (
                        self.data[i].clamp(0, 255) as u8,
                        self.data[i + 1].clamp(0, 255) as u8,
                        self.data[i + 2].clamp(0, 255) as u8,
                    ),
                    ColorSpace::YCbCr => ycbcr_to_rgb(
                        self.data[i].clamp(0, 255),
                        self.data[i + 1].clamp(0, 255),
                        self.data[i + 2].clamp(0, 255),
                    ),
                };
                let a = if has_alpha {
                    self.data[i + 3].clamp(0, 255) as u8
                } else {
                    255
                };
                let o = (y * self.w + x) * 4;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
                out[o + 3] = a;
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct PBC3Result {
    pub image: Vec<u8>,
    pub image_width: u32,
    pub image_height: u32,
    pub data: Vec<u8>,
    pub encode_seconds: f64,
    pub total_bits: u64,
    pub original_width: Option<u32>,
    pub original_height: Option<u32>,
    pub working_width: Option<u32>,
    pub working_height: Option<u32>,
    pub channels: u32,
}

pub struct PBC3;

impl PBC3 {
    fn open_body(data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 6 {
            bail!("not a PBC3 file");
        }
        if data[..4] != MAGIC {
            bail!("not a PBC3 file");
        }
        if data[4] != VERSION {
            bail!("unsupported PBC3 version {}", data[4]);
        }
        entropy_unpack(data[5], &data[6..])
    }

    fn read_header(br: &mut BitReader) -> Result<Header> {
        let downsampled = br.read(1)? != 0;
        let original_w = if downsampled {
            Some(br.read(16)? as usize)
        } else {
            None
        };
        let original_h = if downsampled {
            Some(br.read(16)? as usize)
        } else {
            None
        };
        let w = br.read(16)? as usize;
        let h = br.read(16)? as usize;
        let color_space = ColorSpace::from_id(br.read(2)?)?;
        let channels = br.read(8)? as usize;
        let channel_bits = br.read(4)? as u32;
        let positive_bias = br.read(1)? != 0;
        let has_alpha = br.read(1)? != 0;
        let patch_count = br.read(32)? as usize;
        let mut base_values = Vec::with_capacity(channels);
        for _ in 0..channels {
            base_values.push(br.read(8)? as u8);
        }
        let warmup_on = br.read(1)? != 0;
        let warmup = if warmup_on {
            Some(Warmup {
                w: br.read(16)? as usize,
                h: br.read(16)? as usize,
                split: br.read(32)? as usize,
            })
        } else {
            None
        };
        Ok(Header {
            downsampled,
            original_w,
            original_h,
            w,
            h,
            color_space,
            channels,
            channel_bits,
            positive_bias,
            has_alpha,
            patch_count,
            base_values,
            warmup,
        })
    }

    fn read_patch(br: &mut BitReader, channel_bits: u32, positive_bias: bool) -> Result<Patch> {
        let channel = br.read(channel_bits)? as usize;
        let x = br.read(16)? as usize;
        let y = br.read(16)? as usize;
        let w = br.read(16)? as usize;
        let h = br.read(16)? as usize;
        let pm = br.read(1)? as u8;
        if pm != PALETTE_GENERATED {
            bail!("explicit palette patches were removed in PBC3 3.0 release cleanup");
        }
        let mask_size = br.read(10)? as usize;
        let mut mask = Vec::with_capacity(mask_size);
        for _ in 0..mask_size {
            mask.push(br.read(1)? as u8);
        }
        let negative_max = br.read(8)? as u8;
        let positive_max = br.read(8)? as u8;
        let max_bitcount = br.read(4)? as u8;
        let bitcount = resolve_palette_bitcount(
            &mask,
            max_bitcount,
            negative_max,
            positive_max,
            positive_bias,
        );
        let pal = palette_generator(
            &mask,
            max_bitcount,
            negative_max,
            positive_max,
            positive_bias,
        );
        let cell_size = br.read(16)? as usize;
        let gw = ceil_div(w, cell_size);
        let gh = ceil_div(h, cell_size);
        let mut values = Vec::with_capacity(gw * gh);
        for _ in 0..(gw * gh) {
            values.push(pal[br.read(bitcount as u32)? as usize]);
        }
        Ok(Patch {
            channel,
            x,
            y,
            w,
            h,
            cell_size,
            values,
        })
    }

    fn decode_to_canvas(data: &[u8], max_patches: Option<usize>) -> Result<(Canvas, Header)> {
        let body = Self::open_body(data)?;
        let mut br = BitReader::new(&body);
        let header = Self::read_header(&mut br)?;
        let mut canvas = Canvas::new(header.w, header.h, header.channels, &header.base_values);
        let patches_to_read =
            max_patches.map_or(header.patch_count, |m| min(m, header.patch_count));

        for idx in 0..patches_to_read {
            if let Some(warmup) = &header.warmup {
                if idx == warmup.split {
                    canvas = canvas.resize(warmup.w, warmup.h);
                }
            }
            let patch = Self::read_patch(&mut br, header.channel_bits, header.positive_bias)?;
            canvas.add_patch(&patch);
        }
        Ok((canvas, header))
    }

    pub fn decompress(data: impl AsRef<[u8]>, max_patches: Option<usize>) -> Result<PBC3Result> {
        let t0 = Instant::now();
        let data = data.as_ref();
        let (canvas, header) = Self::decode_to_canvas(data, max_patches)?;
        let mut image = canvas.to_rgba_bytes(header.color_space, header.has_alpha);

        if header.downsampled {
            let original_w = header.original_w.context("missing original width")? as u32;
            let original_h = header.original_h.context("missing original height")? as u32;
            let resized = image::imageops::resize(
                &ImageBuffer::<Rgba<u8>, _>::from_raw(canvas.w as u32, canvas.h as u32, image)
                    .context("invalid RGBA buffer")?,
                original_w,
                original_h,
                FilterType::CatmullRom,
            );
            image = resized.into_raw();
            Ok(PBC3Result {
                image,
                image_width: original_w,
                image_height: original_h,
                data: data.to_vec(),
                encode_seconds: t0.elapsed().as_secs_f64(),
                total_bits: (data.len() as u64) * 8,
                original_width: Some(original_w),
                original_height: Some(original_h),
                working_width: Some(canvas.w as u32),
                working_height: Some(canvas.h as u32),
                channels: header.channels as u32,
            })
        } else {
            Ok(PBC3Result {
                image,
                image_width: canvas.w as u32,
                image_height: canvas.h as u32,
                data: data.to_vec(),
                encode_seconds: t0.elapsed().as_secs_f64(),
                total_bits: (data.len() as u64) * 8,
                original_width: header.original_w.map(|v| v as u32),
                original_height: header.original_h.map(|v| v as u32),
                working_width: Some(canvas.w as u32),
                working_height: Some(canvas.h as u32),
                channels: header.channels as u32,
            })
        }
    }

    pub fn save_png(result: &PBC3Result, path: &Path) -> Result<()> {
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            result.image_width,
            result.image_height,
            result.image.clone(),
        )
        .context("invalid RGBA image")?;
        image.save(path)?;
        Ok(())
    }
}

fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let yf = y as f32;
    let cbf = cb as f32 - 128.0;
    let crf = cr as f32 - 128.0;
    let r = (yf + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
    let g = (yf - 0.344_136 * cbf - 0.714_136 * crf)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (yf + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;
    (r, g, b)
}

fn resize_f32(src: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    if (src_w, src_h) == (dst_w, dst_h) {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; dst_w * dst_h];
    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;
    for y in 0..dst_h {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor().clamp(0.0, (src_h - 1) as f32) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f32;
        for x in 0..dst_w {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor().clamp(0.0, (src_w - 1) as f32) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f32;
            let v00 = src[y0 * src_w + x0];
            let v01 = src[y0 * src_w + x1];
            let v10 = src[y1 * src_w + x0];
            let v11 = src[y1 * src_w + x1];
            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            out[y * dst_w + x] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

fn resize_i16_to_f32(
    src: &[i16],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let src_f: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    resize_f32(&src_f, src_w, src_h, dst_w, dst_h)
}

fn entropy_unpack(method: u8, body: &[u8]) -> Result<Vec<u8>> {
    match method {
        ENTROPY_STORE => Ok(body.to_vec()),
        ENTROPY_LZMA => lzma_raw_lzma2_decompress(body).or_else(|_| python_lzma_decompress(body)),
        _ => bail!("unknown entropy method {}", method),
    }
}

fn python_lzma_decompress(body: &[u8]) -> Result<Vec<u8>> {
    let script = r#"import lzma,sys
data = sys.stdin.buffer.read()
out = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'preset': lzma.PRESET_EXTREME}])
sys.stdout.buffer.write(out)"#;
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn python3 for LZMA fallback")?;
    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        stdin.write_all(body)?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "python LZMA fallback failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output.stdout)
}

fn lzma_raw_lzma2_decompress(input: &[u8]) -> Result<Vec<u8>> {
    let mut out_cap = input.len().saturating_mul(8).max(1024);
    for _ in 0..8 {
        let mut out = vec![0u8; out_cap];
        let mut in_pos: usize = 0;
        let mut out_pos: usize = 0;
        let mut options = unsafe { std::mem::zeroed::<lzma::lzma_options_lzma>() };
        options.dict_size = 1 << 23;
        options.preset_dict = std::ptr::null();
        options.preset_dict_size = 0;
        options.lc = 3;
        options.lp = 0;
        options.pb = 2;
        options.mode = lzma::LZMA_MODE_NORMAL;
        options.nice_len = 64;
        options.mf = lzma::LZMA_MF_BT4;
        options.depth = 0;
        let filters = [
            lzma::lzma_filter {
                id: lzma::LZMA_FILTER_LZMA2,
                options: (&mut options as *mut lzma::lzma_options_lzma).cast::<c_void>(),
            },
            lzma::lzma_filter {
                id: 0,
                options: std::ptr::null_mut::<c_void>(),
            },
        ];
        let ret = unsafe {
            lzma::lzma_raw_buffer_decode(
                filters.as_ptr(),
                std::ptr::null(),
                input.as_ptr(),
                &mut in_pos,
                input.len(),
                out.as_mut_ptr(),
                &mut out_pos,
                out.len(),
            )
        };
        if ret == lzma::LZMA_OK || ret == lzma::LZMA_STREAM_END {
            out.truncate(out_pos);
            return Ok(out);
        }
        if ret == lzma::LZMA_BUF_ERROR {
            out_cap = out_cap.saturating_mul(2);
            continue;
        }
        bail!("lzma decode failed with code {}", ret);
    }
    bail!("lzma decode exceeded retry budget")
}
