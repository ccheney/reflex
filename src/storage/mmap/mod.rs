pub mod config;
pub mod error;

#[cfg(test)]
mod tests;

pub use config::{MmapConfig, MmapMode};
pub use error::{MmapError, MmapResult};

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapMut, MmapOptions as Memmap2Options};
use rkyv::Portable;
use rkyv::api::high::{HighValidator, access};
use rkyv::bytecheck::CheckBytes;
use rkyv::rancor::Error as RkyvError;

pub const RKYV_ALIGNMENT: usize = 16;

enum MmapInner {
    ReadOnly(Mmap),
    Mutable(MmapMut),
}

impl MmapInner {
    fn as_slice(&self) -> &[u8] {
        match self {
            MmapInner::ReadOnly(m) => m.deref(),
            MmapInner::Mutable(m) => m.deref(),
        }
    }

    fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match self {
            MmapInner::ReadOnly(_) => None,
            MmapInner::Mutable(m) => Some(m.as_mut()),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn flush(&self) -> io::Result<()> {
        match self {
            MmapInner::ReadOnly(_) => Ok(()),
            MmapInner::Mutable(m) => m.flush(),
        }
    }

    fn flush_async(&self) -> io::Result<()> {
        match self {
            MmapInner::ReadOnly(_) => Ok(()),
            MmapInner::Mutable(m) => m.flush_async(),
        }
    }

    fn flush_range(&self, offset: usize, len: usize) -> io::Result<()> {
        match self {
            MmapInner::ReadOnly(_) => Ok(()),
            MmapInner::Mutable(m) => m.flush_range(offset, len),
        }
    }
}

pub struct MmapFile {
    mmap: MmapInner,
    file: File,
    config: MmapConfig,
    path: std::path::PathBuf,
}

impl MmapFile {
    pub fn open<P: AsRef<Path>>(path: P, config: MmapConfig) -> MmapResult<Self> {
        let path = path.as_ref();

        let file = match config.mode {
            MmapMode::ReadOnly | MmapMode::CopyOnWrite => {
                OpenOptions::new().read(true).open(path)?
            }
            MmapMode::ReadWrite => OpenOptions::new().read(true).write(true).open(path)?,
        };

        let metadata = file.metadata()?;
        let file_len = metadata.len() as usize;

        if file_len == 0 {
            return Err(MmapError::EmptyFile);
        }

        let mmap = Self::create_mapping(&file, &config)?;

        Ok(Self {
            mmap,
            file,
            config,
            path: path.to_path_buf(),
        })
    }

    pub fn create<P: AsRef<Path>>(
        path: P,
        size: usize,
        mut config: MmapConfig,
    ) -> MmapResult<Self> {
        let path = path.as_ref();
        config.mode = MmapMode::ReadWrite;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(size as u64)?;

        let mmap = Self::create_mapping(&file, &config)?;

        Ok(Self {
            mmap,
            file,
            config,
            path: path.to_path_buf(),
        })
    }

    fn create_mapping(file: &File, config: &MmapConfig) -> MmapResult<MmapInner> {
        let mut opts = Memmap2Options::new();

        if let Some(offset) = config.offset {
            opts.offset(offset);
        }

        if let Some(len) = config.len {
            opts.len(len);
        }

        if config.populate {
            opts.populate();
        }

        let mmap = match config.mode {
            MmapMode::ReadOnly => {
                // SAFETY: We ensure the file exists and is readable.
                // The caller must ensure no concurrent writers modify the file.
                let m = unsafe { opts.map(file)? };
                MmapInner::ReadOnly(m)
            }
            MmapMode::ReadWrite => {
                // SAFETY: We ensure the file exists and is writable.
                // The caller must ensure proper synchronization for concurrent access.
                let m = unsafe { opts.map_mut(file)? };
                MmapInner::Mutable(m)
            }
            MmapMode::CopyOnWrite => {
                // SAFETY: Copy-on-write mappings don't affect the underlying file.
                let m = unsafe { opts.map_copy(file)? };
                MmapInner::Mutable(m)
            }
        };

        Ok(mmap)
    }

    pub fn as_slice(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.mmap.as_mut_slice()
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_writable(&self) -> bool {
        matches!(
            self.config.mode,
            MmapMode::ReadWrite | MmapMode::CopyOnWrite
        )
    }

    pub fn flush(&self) -> MmapResult<()> {
        self.mmap.flush()?;
        Ok(())
    }

    pub fn flush_async(&self) -> MmapResult<()> {
        self.mmap.flush_async()?;
        Ok(())
    }

    pub fn flush_range(&self, offset: usize, len: usize) -> MmapResult<()> {
        self.mmap.flush_range(offset, len)?;
        Ok(())
    }

    pub fn access_archived<T>(&self) -> MmapResult<&T>
    where
        T: Portable + for<'a> CheckBytes<HighValidator<'a, RkyvError>>,
    {
        self.access_archived_at::<T>(0)
    }

    pub fn access_archived_at<T>(&self, offset: usize) -> MmapResult<&T>
    where
        T: Portable + for<'a> CheckBytes<HighValidator<'a, RkyvError>>,
    {
        let data = self.as_slice();

        if offset >= data.len() {
            return Err(MmapError::FileTooSmall {
                expected: offset + 1,
                actual: data.len(),
            });
        }

        let slice = &data[offset..];

        let ptr = slice.as_ptr();
        if !(ptr as usize).is_multiple_of(RKYV_ALIGNMENT) {
            return Err(MmapError::AlignmentError {
                offset,
                alignment: RKYV_ALIGNMENT,
            });
        }

        access::<T, RkyvError>(slice).map_err(|e| MmapError::ValidationFailed(format!("{:?}", e)))
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        self.as_mut_slice().map(|s| s.as_mut_ptr())
    }

    pub fn grow(&mut self, new_size: usize) -> MmapResult<()> {
        if self.config.mode == MmapMode::ReadOnly {
            return Err(MmapError::ResizeFailed(
                "Cannot grow read-only mapping".to_string(),
            ));
        }

        let current_size = self.len();
        if new_size <= current_size {
            return Err(MmapError::ResizeFailed(format!(
                "New size {} must be larger than current size {}",
                new_size, current_size
            )));
        }

        self.flush()?;

        self.file.set_len(new_size as u64)?;

        self.mmap = Self::create_mapping(&self.file, &self.config)?;

        Ok(())
    }

    pub fn shrink(&mut self, new_size: usize) -> MmapResult<()> {
        if self.config.mode == MmapMode::ReadOnly {
            return Err(MmapError::ResizeFailed(
                "Cannot shrink read-only mapping".to_string(),
            ));
        }

        if new_size == 0 {
            return Err(MmapError::ResizeFailed(
                "Cannot shrink to zero size".to_string(),
            ));
        }

        let current_size = self.len();
        if new_size >= current_size {
            return Err(MmapError::ResizeFailed(format!(
                "New size {} must be smaller than current size {}",
                new_size, current_size
            )));
        }

        self.flush()?;

        self.file.set_len(new_size as u64)?;

        self.mmap = Self::create_mapping(&self.file, &self.config)?;

        Ok(())
    }

    pub fn resize(&mut self, new_size: usize) -> MmapResult<()> {
        let current_size = self.len();

        if new_size > current_size {
            self.grow(new_size)
        } else if new_size < current_size {
            self.shrink(new_size)
        } else {
            Ok(())
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mode(&self) -> MmapMode {
        self.config.mode
    }
}

#[derive(Clone)]
pub struct MmapFileHandle {
    inner: Arc<Mmap>,
    path: Arc<std::path::PathBuf>,
}

impl std::fmt::Debug for MmapFileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapFileHandle")
            .field("path", &self.path)
            .field("len", &self.len())
            .field("strong_count", &self.strong_count())
            .finish()
    }
}

impl MmapFileHandle {
    pub fn open<P: AsRef<Path>>(path: P) -> MmapResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;

        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            return Err(MmapError::EmptyFile);
        }

        // SAFETY: We ensure the file exists and is readable.
        // The Arc wrapper provides thread-safe shared access.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            inner: Arc::new(mmap),
            path: Arc::new(path.to_path_buf()),
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        self.inner.deref()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    pub fn access_archived<T>(&self) -> MmapResult<&T>
    where
        T: Portable + for<'a> CheckBytes<HighValidator<'a, RkyvError>>,
    {
        self.access_archived_at::<T>(0)
    }

    pub fn access_archived_at<T>(&self, offset: usize) -> MmapResult<&T>
    where
        T: Portable + for<'a> CheckBytes<HighValidator<'a, RkyvError>>,
    {
        let data = self.as_slice();

        if offset >= data.len() {
            return Err(MmapError::FileTooSmall {
                expected: offset + 1,
                actual: data.len(),
            });
        }

        let slice = &data[offset..];

        let ptr = slice.as_ptr();
        if !(ptr as usize).is_multiple_of(RKYV_ALIGNMENT) {
            return Err(MmapError::AlignmentError {
                offset,
                alignment: RKYV_ALIGNMENT,
            });
        }

        access::<T, RkyvError>(slice).map_err(|e| MmapError::ValidationFailed(format!("{:?}", e)))
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub struct AlignedMmapBuilder {
    path: std::path::PathBuf,
}

impl AlignedMmapBuilder {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn write(self, data: &[u8]) -> MmapResult<MmapFile> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        file.write_all(data)?;
        file.flush()?;
        drop(file);

        MmapFile::open(&self.path, MmapConfig::read_write())
    }

    pub fn write_readonly(self, data: &[u8]) -> MmapResult<MmapFileHandle> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        file.write_all(data)?;
        file.flush()?;
        drop(file);

        MmapFileHandle::open(&self.path)
    }
}
