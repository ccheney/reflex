#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapMode {
    ReadOnly,
    ReadWrite,
    CopyOnWrite,
}

#[derive(Debug, Clone)]
pub struct MmapConfig {
    pub mode: MmapMode,
    pub populate: bool,
    pub offset: Option<u64>,
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
    pub fn read_only() -> Self {
        Self {
            mode: MmapMode::ReadOnly,
            ..Default::default()
        }
    }

    pub fn read_write() -> Self {
        Self {
            mode: MmapMode::ReadWrite,
            ..Default::default()
        }
    }

    pub fn copy_on_write() -> Self {
        Self {
            mode: MmapMode::CopyOnWrite,
            ..Default::default()
        }
    }

    pub fn with_populate(mut self) -> Self {
        self.populate = true;
        self
    }

    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_len(mut self, len: usize) -> Self {
        self.len = Some(len);
        self
    }
}
