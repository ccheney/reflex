use bitvec::prelude::*;

pub fn quantize_to_binary(vector: &[f32]) -> Vec<u8> {
    let mut bv = BitVec::<u8, Lsb0>::with_capacity(vector.len());
    for &val in vector {
        bv.push(val > 0.0);
    }
    bv.into_vec()
}

pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    if a.len() != b.len() {
        return u32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}
