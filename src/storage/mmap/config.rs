#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Memory-mapping mode.
pub enum MmapMode {
    /// Read-only mapping.
    ReadOnly,
    /// Read/write mapping.
    ReadWrite,
    /// Copy-on-write mapping (writes do not hit disk).
    CopyOnWrite,
}

#[derive(Debug, Clone)]
/// Options for creating a memory map.
pub struct MmapConfig {
    /// Mapping mode.
    pub mode: MmapMode,
    /// If true, ask the OS to populate pages eagerly.
    pub populate: bool,
    /// Optional file offset (bytes).
    pub offset: Option<u64>,
    /// Optional length (bytes).
    pub len: Option<usize>,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            mode: MmapMode::ReadOnly,
            populate: false,
            offset: None,
            len: None,
        }
    }
}

impl MmapConfig {
    /// Read-only config.
    pub fn read_only() -> Self {
        Self {
            mode: MmapMode::ReadOnly,
            ..Default::default()
        }
    }

    /// Read/write config.
    pub fn read_write() -> Self {
        Self {
            mode: MmapMode::ReadWrite,
            ..Default::default()
        }
    }

    /// Copy-on-write config.
    pub fn copy_on_write() -> Self {
        Self {
            mode: MmapMode::CopyOnWrite,
            ..Default::default()
        }
    }

    /// Enables eager page population.
    pub fn with_populate(mut self) -> Self {
        self.populate = true;
        self
    }

    /// Sets an offset (bytes).
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Sets a length (bytes).
    pub fn with_len(mut self, len: usize) -> Self {
        self.len = Some(len);
        self
    }
}
