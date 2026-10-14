use crate::npu::QuantMatrix;
use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions, UncheckedAdvice};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    ops::Range,
    path::{Component, Path},
    sync::Arc,
};

#[derive(Clone)]
pub struct Tensor {
    map: Arc<Mmap>,
    range: Range<usize>,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn data(&self) -> &[u8] {
        &self.map[self.range.clone()]
    }
    pub fn require_shape(&self, shape: &[usize], name: &str) -> Result<()> {
        ensure!(
            self.shape == shape,
            "unexpected shape: {name}: {:?}, expected {shape:?}",
            self.shape
        );
        Ok(())
    }
    pub fn floating(&self) -> bool {
        matches!(self.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
    }
    pub fn number(&self, i: usize) -> f32 {
        let b = self.data();
        match self.dtype {
            Dtype::F32 => f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()),
            Dtype::F16 => {
                f16::from_bits(u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap())).to_f32()
            }
            Dtype::BF16 => {
                bf16::from_bits(u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap()))
                    .to_f32()
            }
            _ => unreachable!("validated floating tensor"),
        }
    }
    pub fn integer(&self, i: usize) -> Result<i64> {
        let b = self.data();
        Ok(match self.dtype {
            Dtype::I8 => b[i] as i8 as i64,
            Dtype::I32 => i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as i64,
            Dtype::I64 => i64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()),
            _ => bail!("expected integer tensor"),
        })
    }
    pub fn floats(&self) -> Result<Vec<f32>> {
        ensure!(self.floating(), "expected floating tensor");
        let values: Vec<_> = (0..self.shape.iter().product())
            .map(|i| self.number(i))
            .collect();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite model tensor"
        );
        Ok(values)
    }
    pub fn row(&self, row: usize, output: &mut [f32]) {
        let width = self.shape[1];
        let mut i = 0;
        #[cfg(target_arch = "aarch64")]
        if self.dtype == Dtype::BF16 {
            unsafe {
                use std::arch::aarch64::*;
                let pointer = self.data().as_ptr().cast::<u16>().add(row * width);
                while i + 8 <= output.len() {
                    let values = vld1q_u16(pointer.add(i));
                    vst1q_f32(
                        output.as_mut_ptr().add(i),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(values))),
                    );
                    vst1q_f32(
                        output.as_mut_ptr().add(i + 4),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(values))),
                    );
                    i += 8;
                }
            }
        }
        while i < output.len() {
            output[i] = self.number(row * width + i);
            i += 1;
        }
    }
}

pub struct TensorStore {
    tensors: HashMap<String, Tensor>,
    maps: Vec<Arc<Mmap>>,
}

impl TensorStore {
    pub fn open(directory: &Path) -> Result<Self> {
        let mut store = Self {
            tensors: HashMap::new(),
            maps: Vec::new(),
        };
        let index_path = directory.join("model.safetensors.index.json");
        if index_path.exists() {
            let index: Value = serde_json::from_slice(&std::fs::read(index_path)?)?;
            let weights = index["weight_map"]
                .as_object()
                .context("missing safetensors weight_map")?;
            let mut shards = BTreeSet::new();
            for value in weights.values() {
                let name = value.as_str().context("invalid shard filename")?;
                let mut components = Path::new(name).components();
                ensure!(
                    matches!(components.next(), Some(Component::Normal(_)))
                        && components.next().is_none(),
                    "shard index requires plain filenames"
                );
                shards.insert(name.to_owned());
            }
            for shard in shards {
                store.add_file(&directory.join(&shard), Some(weights))?;
            }
            ensure!(
                store.tensors.len() == weights.len(),
                "index has missing tensors"
            );
        } else {
            store.add_file(&directory.join("model.safetensors"), None)?;
        }
        Ok(store)
    }
    fn add_file(
        &mut self,
        path: &Path,
        index: Option<&serde_json::Map<String, Value>>,
    ) -> Result<()> {
        let file =
            File::open(path).with_context(|| format!("cannot open shard {}", path.display()))?;
        ensure!(file.metadata()?.is_file(), "invalid safetensors shard");
        // Checkpoint files stay immutable while the model is alive.
        let map = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        let tensors = SafeTensors::deserialize(&map)
            .with_context(|| format!("invalid safetensors shard {}", path.display()))?;
        for (name, view) in tensors.tensors() {
            if let Some(index) = index {
                ensure!(
                    index.get(&name).and_then(Value::as_str)
                        == path.file_name().and_then(|p| p.to_str()),
                    "tensor/index mismatch: {name}"
                );
            }
            ensure!(
                matches!(
                    view.dtype(),
                    Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::I8 | Dtype::I32 | Dtype::I64
                ),
                "unsupported safetensors dtype: {:?}",
                view.dtype()
            );
            let start = view.data().as_ptr() as usize - map.as_ptr() as usize;
            let tensor = Tensor {
                map: map.clone(),
                range: start..start + view.data().len(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
            };
            ensure!(
                self.tensors.insert(name.clone(), tensor).is_none(),
                "duplicate tensor: {name}"
            );
        }
        drop(tensors);
        self.maps.push(map);
        Ok(())
    }
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .with_context(|| format!("missing tensor: {name}"))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn mapped_bytes(&self) -> usize {
        self.maps.iter().map(|m| m.len()).sum()
    }
    pub fn release_pages(&self) -> Result<()> {
        for map in &self.maps {
            unsafe {
                map.unchecked_advise(UncheckedAdvice::DontNeed)?;
            }
        }
        Ok(())
    }
}

pub struct QuantLoader {
    pub store: TensorStore,
    quant: Value,
}

fn matches(target: &str, module: &str) -> Result<bool> {
    if target == "Linear" || target == module {
        return Ok(true);
    }
    if let Some(pattern) = target.strip_prefix("re:") {
        return Ok(regex::Regex::new(&format!("^(?:{pattern})$"))?.is_match(module));
    }
    Ok(false)
}
fn symmetric(q: &Value, bits: u32, strategy: &str, dynamic: bool) -> Result<()> {
    ensure!(
        q["num_bits"].as_u64() == Some(bits as u64)
            && q["type"].as_str() == Some("int")
            && q["strategy"].as_str() == Some(strategy)
            && q["symmetric"].as_bool() == Some(true)
            && q.get("dynamic")
                .map_or(!dynamic, |v| v.as_bool() == Some(dynamic))
            && q["group_size"].is_null()
            && q["actorder"].is_null(),
        "expected symmetric static channel W4/W8 and dynamic token A8; unsupported group scales/actorder"
    );
    Ok(())
}

fn transpose_codes(codes: &mut [i8], n: usize, read: impl Fn(usize, usize) -> i8 + Sync) {
    codes
        .par_chunks_mut(n * 32)
        .enumerate()
        .for_each(|(block_index, block)| {
            for n0 in (0..n).step_by(64) {
                for (local_k, row) in block.chunks_mut(n).enumerate() {
                    let k = block_index * 32 + local_k;
                    for (ni, value) in row.iter_mut().enumerate().take((n0 + 64).min(n)).skip(n0) {
                        *value = read(k, ni);
                    }
                }
            }
        });
}

impl QuantLoader {
    pub fn new(store: TensorStore, config: &Value) -> Result<Self> {
        let quant = config["quantization_config"].clone();
        ensure!(
            quant["quant_method"] == "compressed-tensors"
                && quant["quantization_status"] == "compressed",
            "expected a compressed-tensors compressed checkpoint"
        );
        for key in ["sparsity_config", "transform_config"] {
            let v = &quant[key];
            ensure!(
                v.is_null() || v.as_object().is_some_and(|v| v.is_empty()),
                "unsupported compressed-tensors {key}"
            );
        }
        ensure!(
            quant["kv_cache_scheme"].is_null(),
            "quantized KV cache schemes are unsupported"
        );
        ensure!(
            quant["config_groups"].is_object(),
            "missing quantization config_groups"
        );
        Ok(Self { store, quant })
    }
    fn scheme(&self, module: &str) -> Result<&Value> {
        if let Some(ignore) = self.quant["ignore"].as_array() {
            for target in ignore {
                ensure!(
                    !matches(target.as_str().context("invalid ignore target")?, module)?,
                    "required projection excluded from quantization: {module}"
                );
            }
        }
        let mut found = None;
        for value in self.quant["config_groups"].as_object().unwrap().values() {
            for target in value["targets"]
                .as_array()
                .context("missing quantization targets")?
            {
                if matches(
                    target.as_str().context("invalid quantization target")?,
                    module,
                )? {
                    if let Some(previous) = found {
                        ensure!(previous == value, "ambiguous quantization scheme: {module}");
                    }
                    found = Some(value);
                }
            }
        }
        found.with_context(|| format!("no quantization scheme for {module}"))
    }
    pub fn bits(&self, module: &str) -> Result<u32> {
        let s = self.scheme(module)?;
        let bits = s["weights"]["num_bits"]
            .as_u64()
            .context("missing weight bits")? as u32;
        ensure!(bits == 4 || bits == 8, "only W4A8 and W8A8 are supported");
        symmetric(&s["weights"], bits, "channel", false)?;
        symmetric(&s["input_activations"], 8, "token", true)?;
        ensure!(
            s["output_activations"].is_null(),
            "quantized linear output is unsupported: {module}"
        );
        Ok(bits)
    }
    pub fn load(&self, module: &str, n: usize, k: usize) -> Result<QuantMatrix> {
        ensure!(n > 0 && k > 0, "invalid linear dimensions");
        let bits = self.bits(module)?;
        ensure!(
            !self.store.contains(&format!("{module}.bias")),
            "linear bias is unsupported: {module}"
        );
        ensure!(
            !self.store.contains(&format!("{module}.weight_g_idx")),
            "weight_g_idx/actorder unsupported: {module}"
        );
        let scale = self.store.get(&format!("{module}.weight_scale"))?;
        scale.require_shape(&[n, 1], "weight_scale")?;
        let scales = scale.floats()?;
        ensure!(
            scales.iter().all(|v| *v > 0.0),
            "nonpositive scale: {module}"
        );
        if self.store.contains(&format!("{module}.weight_zero_point")) {
            let zero = self.store.get(&format!("{module}.weight_zero_point"))?;
            for i in 0..zero.shape.iter().product() {
                ensure!(zero.integer(i)? == 0, "nonzero weight zero point: {module}");
            }
        }
        let scheme = self.scheme(module)?;
        let format = scheme
            .get("format")
            .or_else(|| self.quant.get("format"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let mut codes = vec![0i8; k.checked_mul(n).context("weight size overflow")?];
        match format {
            "pack-quantized" => {
                let packed = self.store.get(&format!("{module}.weight_packed"))?;
                let factor = 32 / bits as usize;
                let columns = k.div_ceil(factor);
                packed.require_shape(&[n, columns], "weight_packed")?;
                ensure!(packed.dtype == Dtype::I32, "weight_packed must be I32");
                let shape = self.store.get(&format!("{module}.weight_shape"))?;
                shape.require_shape(&[2], "weight_shape")?;
                ensure!(
                    shape.integer(0)? == n as i64 && shape.integer(1)? == k as i64,
                    "weight_shape mismatch: {module}"
                );
                let data = packed.data();
                transpose_codes(&mut codes, n, |ki, ni| {
                    let offset = (ni * columns + ki / factor) * 4;
                    let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    (((word >> ((ki % factor) * bits as usize)) & ((1 << bits) - 1)) as i32
                        - (1i32 << (bits - 1))) as i8
                });
            }
            "int-quantized" if bits == 8 => {
                let weights = self.store.get(&format!("{module}.weight"))?;
                weights.require_shape(&[n, k], "weight")?;
                ensure!(weights.dtype == Dtype::I8, "W8 weight must be signed I8");
                let data = weights.data();
                transpose_codes(&mut codes, n, |ki, ni| data[ni * k + ki] as i8);
            }
            _ => bail!("unsupported compressed-tensors storage: {format}"),
        }
        self.store.release_pages()?;
        Ok(QuantMatrix {
            bits,
            k,
            n,
            codes,
            scales,
        })
    }
}

pub fn concatenate(mut matrices: Vec<QuantMatrix>) -> Result<QuantMatrix> {
    ensure!(!matrices.is_empty(), "empty fused projection");
    if matrices.len() == 1 {
        return Ok(matrices.pop().unwrap());
    }
    let bits = matrices[0].bits;
    let k = matrices[0].k;
    ensure!(
        matrices.iter().all(|m| m.bits == bits && m.k == k),
        "fused projections require identical bit widths and K"
    );
    let n = matrices.iter().map(|m| m.n).sum();
    let scales = matrices
        .iter()
        .flat_map(|m| m.scales.iter().copied())
        .collect();
    let mut codes = vec![0; k * n];
    codes.par_chunks_mut(n).enumerate().for_each(|(ki, row)| {
        let mut offset = 0;
        for m in &matrices {
            row[offset..offset + m.n].copy_from_slice(&m.codes[ki * m.n..(ki + 1) * m.n]);
            offset += m.n;
        }
    });
    Ok(QuantMatrix {
        bits,
        k,
        n,
        codes,
        scales,
    })
}
