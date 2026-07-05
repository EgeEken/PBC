use std::cmp::{max, min};

pub fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn range_counts(
    mask_size: usize,
    negative_max: u8,
    positive_max: u8,
    positive_bias: bool,
) -> (usize, usize) {
    let side_bits = mask_size.saturating_sub(1);
    let negative_max = negative_max as usize;
    let positive_max = positive_max as usize;
    if side_bits == 0 || (negative_max == 0 && positive_max == 0) {
        return (0, 0);
    }
    if negative_max == 0 {
        return (min(side_bits, positive_max), 0);
    }
    if positive_max == 0 {
        return (0, min(side_bits, negative_max));
    }

    let raw_num = side_bits * positive_max;
    let raw_den = positive_max + negative_max;
    let pos_count = if positive_bias {
        raw_num.div_ceil(raw_den)
    } else {
        raw_num / raw_den
    };
    let mut pos_count = min(side_bits - 1, max(1, pos_count));
    pos_count = min(pos_count, positive_max);
    let mut neg_count = min(side_bits - pos_count, negative_max);
    if neg_count == 0 && negative_max > 0 && side_bits > pos_count {
        neg_count = 1;
        pos_count = max(1, pos_count - 1);
    }
    (pos_count, neg_count)
}

fn range_for_mask_index(
    index: usize,
    mask_size: usize,
    negative_max: u8,
    positive_max: u8,
    positive_bias: bool,
) -> Option<(i16, i16)> {
    let (pos_count, neg_count) = range_counts(mask_size, negative_max, positive_max, positive_bias);
    let negative_max = negative_max as usize;
    let positive_max = positive_max as usize;
    if index == 0 {
        return Some((0, 0));
    }
    if (1..=pos_count).contains(&index) {
        let bin_i = index - 1;
        let start = 1 + (bin_i * positive_max) / pos_count;
        let end = ((bin_i + 1) * positive_max) / pos_count;
        return (start <= end).then_some((start as i16, end as i16));
    }
    let bin_i = index - 1 - pos_count;
    if bin_i < neg_count {
        let low_mag = 1 + (bin_i * negative_max) / neg_count;
        let high_mag = ((bin_i + 1) * negative_max) / neg_count;
        return (high_mag >= low_mag).then_some((-(high_mag as i16), -(low_mag as i16)));
    }
    None
}

fn active_value_count(
    mask: &[u8],
    negative_max: u8,
    positive_max: u8,
    positive_bias: bool,
) -> usize {
    let mut count = 0usize;
    for (i, bit) in mask.iter().enumerate() {
        if *bit != 0 {
            if let Some((start, end)) =
                range_for_mask_index(i, mask.len(), negative_max, positive_max, positive_bias)
            {
                count += (end - start + 1) as usize;
            }
        }
    }
    max(1, count)
}

pub fn resolve_palette_bitcount(
    mask: &[u8],
    max_bitcount: u8,
    negative_max: u8,
    positive_max: u8,
    positive_bias: bool,
) -> u8 {
    let value_count = active_value_count(mask, negative_max, positive_max, positive_bias);
    let needed = ((usize::BITS - (value_count - 1).leading_zeros()) as u8).max(1);
    min(needed, max_bitcount)
}

pub fn palette_generator(
    mask: &[u8],
    max_bitcount: u8,
    negative_max: u8,
    positive_max: u8,
    positive_bias: bool,
) -> Vec<i16> {
    let bitcount = resolve_palette_bitcount(
        mask,
        max_bitcount,
        negative_max,
        positive_max,
        positive_bias,
    );
    let size = 1usize << bitcount;
    let mut active_ranges = Vec::new();
    for (i, bit) in mask.iter().enumerate() {
        if *bit != 0 {
            if let Some(range) =
                range_for_mask_index(i, mask.len(), negative_max, positive_max, positive_bias)
            {
                active_ranges.push(range);
            }
        }
    }

    let mut palette = Vec::new();
    if mask.first().copied().unwrap_or(0) != 0 {
        palette.push(0);
        active_ranges.retain(|r| *r != (0, 0));
    }

    let value_count = active_value_count(mask, negative_max, positive_max, positive_bias);
    if size >= value_count {
        for (start, end) in active_ranges {
            for value in start..=end {
                palette.push(value);
            }
        }
        while palette.len() < size {
            palette.push(*palette.last().unwrap_or(&0));
        }
        palette.truncate(size);
        return palette;
    }

    if active_ranges.is_empty() {
        return vec![0; size];
    }

    let remaining = size - palette.len();
    let mut counts = vec![0usize; active_ranges.len()];
    for i in 0..remaining {
        counts[i % active_ranges.len()] += 1;
    }
    for ((start, end), count) in active_ranges.into_iter().zip(counts) {
        if count == 1 {
            palette.push(((start as f64 + end as f64) / 2.0).round() as i16);
        } else if count > 1 {
            for j in 0..count {
                let t = (j + 1) as f64 / (count + 1) as f64;
                palette.push((start as f64 + (end - start) as f64 * t).round() as i16);
            }
        }
    }
    while palette.len() < size {
        palette.push(*palette.last().unwrap_or(&0));
    }
    palette.truncate(size);
    palette
}
