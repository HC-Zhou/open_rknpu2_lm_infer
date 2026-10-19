//! CPU image preparation and reusable vision arithmetic. No device access.
pub mod math;
mod processor;
mod resize;
pub use processor::{ImageProcessor, MAX_IMAGES, PreparedImage, data_url};
pub(crate) fn processor_limit() -> usize {
    processor::MAX_REQUEST_IMAGE_BYTES
}
