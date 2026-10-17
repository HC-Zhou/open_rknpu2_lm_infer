/// W8 assigns at least one 32-column N tile to every active core.
/// Keep the requested mask for large projections and select a legal subset for
/// the narrow GDN gate projections.
pub(super) fn w8_active_mask(requested: u32, columns: usize) -> u32 {
    let groups = columns.div_ceil(32);
    if groups >= requested.count_ones() as usize {
        requested
    } else if groups >= 2 && requested & 3 == 3 {
        3
    } else {
        requested & requested.wrapping_neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn narrow_w8_preserves_requested_cores_when_there_are_enough_tiles() {
        for mask in [1u32, 2, 4, 3, 7] {
            for n in [1, 31, 32, 33, 64, 65, 256, 248320] {
                let active = w8_active_mask(mask, n);
                assert_ne!(active, 0);
                assert_eq!(active & mask, active);
                assert!(active.count_ones() as usize <= n.div_ceil(32));
                if n.div_ceil(32) >= mask.count_ones() as usize {
                    assert_eq!(active, mask);
                }
            }
        }
        assert_eq!(w8_active_mask(7, 64), 3);
        assert_eq!(w8_active_mask(7, 32), 1);
        assert_eq!(w8_active_mask(4, 32), 4);
    }
}
