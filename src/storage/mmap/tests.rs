use super::*;
use rkyv::rancor::Error;
use rkyv::to_bytes;
use std::io::Write;
use tempfile::NamedTempFile;

use crate::storage::CacheEntry;

const TEST_MMAP_LEN: usize = 1024;

fn create_test_file(content: &[u8]) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    file.write_all(content)
        .expect("Failed to write to temp file");
    file.flush().expect("Failed to flush temp file");
    file
}

fn create_serialized_entry() -> Vec<u8> {
    let entry = CacheEntry {
        tenant_id: 12345,
        context_hash: 67890,
        timestamp: 1702500000,
        embedding: vec![0x01, 0x02, 0x03, 0x04],
        payload_blob: b"test payload".to_vec(),
    };

    to_bytes::<Error>(&entry)
        .expect("Failed to serialize")
        .into_vec()
}

#[test]
fn test_open_read_only() {
    let content = b"hello, world!";
    let file = create_test_file(content);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    assert_eq!(mmap.as_slice(), content);
    assert_eq!(mmap.len(), content.len());
    assert!(!mmap.is_empty());
    assert!(!mmap.is_writable());
    assert_eq!(mmap.mode(), MmapMode::ReadOnly);
}

#[test]
fn test_open_read_write() {
    let content = b"hello, world!";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    assert!(mmap.is_writable());
    assert_eq!(mmap.mode(), MmapMode::ReadWrite);

    let slice = mmap.as_mut_slice().expect("Should be writable");
    slice[0] = b'H';

    assert_eq!(&mmap.as_slice()[..5], b"Hello");
}

#[test]
fn test_open_copy_on_write() {
    let content = b"hello, world!";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::copy_on_write()).expect("Failed to open mmap");

    assert!(mmap.is_writable());
    assert_eq!(mmap.mode(), MmapMode::CopyOnWrite);

    let slice = mmap.as_mut_slice().expect("Should be writable");
    slice[0] = b'H';

    let original = std::fs::read(file.path()).expect("Failed to read file");
    assert_eq!(&original[..5], b"hello");
}

#[test]
fn test_open_empty_file_fails() {
    let file = NamedTempFile::new().expect("Failed to create temp file");

    let result = MmapFile::open(file.path(), MmapConfig::read_only());

    assert!(matches!(result, Err(MmapError::EmptyFile)));
}

#[test]
fn test_open_nonexistent_file_fails() {
    let result = MmapFile::open("/nonexistent/path/to/file", MmapConfig::read_only());

    assert!(matches!(result, Err(MmapError::Io(_))));
}

#[test]
fn test_create_file() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("new_file.bin");

    let mmap = MmapFile::create(&path, TEST_MMAP_LEN, MmapConfig::default())
        .expect("Failed to create mmap");

    assert_eq!(mmap.len(), TEST_MMAP_LEN);
    assert!(mmap.is_writable());
    assert!(path.exists());
}

#[test]
fn test_create_and_write() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("written_file.bin");

    let mut mmap =
        MmapFile::create(&path, 100, MmapConfig::default()).expect("Failed to create mmap");

    let data = b"test data here";
    let slice = mmap.as_mut_slice().expect("Should be writable");
    slice[..data.len()].copy_from_slice(data);

    mmap.flush().expect("Failed to flush");

    let read_mmap = MmapFile::open(&path, MmapConfig::read_only()).expect("Failed to open");
    assert_eq!(&read_mmap.as_slice()[..data.len()], data);
}

#[test]
fn test_grow_file() {
    let content = b"initial content";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    let original_len = mmap.len();
    mmap.grow(original_len * 2).expect("Failed to grow");

    assert_eq!(mmap.len(), original_len * 2);
    assert_eq!(&mmap.as_slice()[..content.len()], content);
}

#[test]
fn test_shrink_file() {
    let content = b"initial content that is longer";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    mmap.shrink(10).expect("Failed to shrink");

    assert_eq!(mmap.len(), 10);
    assert_eq!(mmap.as_slice(), &content[..10]);
}

#[test]
fn test_resize_grow() {
    let content = b"content";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    mmap.resize(100).expect("Failed to resize");
    assert_eq!(mmap.len(), 100);
}

#[test]
fn test_resize_shrink() {
    let content = b"longer content here";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    mmap.resize(5).expect("Failed to resize");
    assert_eq!(mmap.len(), 5);
}

#[test]
fn test_resize_same_size() {
    let content = b"content";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    let len = mmap.len();
    mmap.resize(len).expect("Failed to resize");
    assert_eq!(mmap.len(), len);
}

#[test]
fn test_grow_readonly_fails() {
    let content = b"content";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    let result = mmap.grow(100);
    assert!(matches!(result, Err(MmapError::ResizeFailed(_))));
}

#[test]
fn test_shrink_to_zero_fails() {
    let content = b"content";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    let result = mmap.shrink(0);
    assert!(matches!(result, Err(MmapError::ResizeFailed(_))));
}

#[test]
fn test_access_archived_cache_entry() {
    use crate::storage::ArchivedCacheEntry;

    let data = create_serialized_entry();
    let file = create_test_file(&data);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    let archived = mmap
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access archived");

    assert_eq!(archived.tenant_id, 12345);
    assert_eq!(archived.context_hash, 67890);
    assert_eq!(archived.timestamp, 1702500000);
}

#[test]
fn test_handle_open() {
    let content = b"handle test content";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    assert_eq!(handle.as_slice(), content);
    assert_eq!(handle.len(), content.len());
    assert!(!handle.is_empty());
}

#[test]
fn test_handle_clone() {
    let content = b"cloned content";
    let file = create_test_file(content);

    let handle1 = MmapFileHandle::open(file.path()).expect("Failed to open handle");
    let handle2 = handle1.clone();

    assert_eq!(handle1.as_slice(), handle2.as_slice());
    assert_eq!(handle1.strong_count(), 2);
    assert_eq!(handle2.strong_count(), 2);
}

#[test]
fn test_handle_thread_safety() {
    let content = b"thread safe content";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    let handle_clone = handle.clone();

    let thread = std::thread::spawn(move || {
        assert_eq!(handle_clone.as_slice(), content);
        handle_clone.len()
    });

    let len = thread.join().expect("Thread panicked");
    assert_eq!(len, content.len());
}

#[test]
fn test_handle_concurrent_access() {
    let content = b"concurrent access test data";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    let mut threads = vec![];

    for _ in 0..10 {
        let h = handle.clone();
        threads.push(std::thread::spawn(move || {
            for _ in 0..100 {
                let _ = h.as_slice();
                let _ = h.len();
            }
        }));
    }

    for t in threads {
        t.join().expect("Thread panicked");
    }

    assert!(handle.strong_count() == 1);
}

#[test]
fn test_aligned_builder_write() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("aligned.bin");

    let data = b"test data";
    let builder = AlignedMmapBuilder::new(&path);

    let mmap = builder.write(data).expect("Failed to write");

    assert_eq!(mmap.len(), data.len());
    assert_eq!(mmap.as_slice(), data);
}

#[test]
fn test_aligned_builder_write_readonly() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("aligned_ro.bin");

    let data = b"readonly test data";
    let builder = AlignedMmapBuilder::new(&path);

    let handle = builder.write_readonly(data).expect("Failed to write");

    assert_eq!(handle.len(), data.len());
    assert_eq!(handle.as_slice(), data);
}

#[test]
fn test_aligned_builder_rkyv_entry() {
    use crate::storage::ArchivedCacheEntry;

    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("entry.rkyv");

    let data = create_serialized_entry();

    let file = create_test_file(&data);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access archived");

    assert_eq!(archived.tenant_id, 12345);

    let builder = AlignedMmapBuilder::new(&path);
    let mmap = builder.write(&data).expect("Failed to write");

    let archived2 = mmap
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access archived via MmapFile");

    assert_eq!(archived2.tenant_id, 12345);
}

#[test]
fn test_flush_operations() {
    let content = b"flush test";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    mmap.as_mut_slice().unwrap()[0] = b'F';

    mmap.flush().expect("flush failed");
    mmap.flush_async().expect("flush_async failed");
    mmap.flush_range(0, 1).expect("flush_range failed");
}

#[test]
fn test_readonly_flush_is_noop() {
    let content = b"readonly flush";
    let file = create_test_file(content);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    mmap.flush().expect("flush failed");
    mmap.flush_async().expect("flush_async failed");
    mmap.flush_range(0, 1).expect("flush_range failed");
}

#[test]
fn test_config_builder_pattern() {
    let config = MmapConfig::read_only()
        .with_populate()
        .with_offset(4096)
        .with_len(TEST_MMAP_LEN);

    assert_eq!(config.mode, MmapMode::ReadOnly);
    assert!(config.populate);
    assert_eq!(config.offset, Some(4096));
    assert_eq!(config.len, Some(TEST_MMAP_LEN));
}

#[test]
fn test_path_accessor() {
    let content = b"path test";
    let file = create_test_file(content);
    let expected_path = file.path().to_path_buf();

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    assert_eq!(mmap.path(), expected_path);
}

#[test]
fn test_handle_path_accessor() {
    let content = b"handle path test";
    let file = create_test_file(content);
    let expected_path = file.path().to_path_buf();

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    assert_eq!(handle.path(), expected_path);
}

#[test]
fn test_as_ptr() {
    let content = b"pointer test";
    let file = create_test_file(content);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    let ptr = mmap.as_ptr();
    assert!(!ptr.is_null());

    unsafe {
        assert_eq!(*ptr, b'p');
    }
}

#[test]
fn test_as_mut_ptr() {
    let content = b"mut pointer test";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    let ptr = mmap.as_mut_ptr().expect("Should have mut ptr");
    assert!(!ptr.is_null());

    unsafe {
        *ptr = b'M';
    }

    assert_eq!(mmap.as_slice()[0], b'M');
}

#[test]
fn test_readonly_has_no_mut_ptr() {
    let content = b"readonly no mut";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    assert!(mmap.as_mut_ptr().is_none());
}

#[test]
fn test_handle_as_ptr() {
    let content = b"handle pointer";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    let ptr = handle.as_ptr();
    assert!(!ptr.is_null());
}

#[test]
fn test_handle_open_empty_file_fails() {
    let file = NamedTempFile::new().expect("Failed to create temp file");

    let result = MmapFileHandle::open(file.path());

    assert!(matches!(result, Err(MmapError::EmptyFile)));
}

#[test]
fn test_handle_debug_impl() {
    let content = b"debug test";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");
    let debug_str = format!("{:?}", handle);

    assert!(debug_str.contains("MmapFileHandle"));
    assert!(debug_str.contains("len"));
    assert!(debug_str.contains("strong_count"));
}

#[test]
fn test_access_archived_at_offset_too_large() {
    use crate::storage::ArchivedCacheEntry;

    let data = create_serialized_entry();
    let file = create_test_file(&data);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    // Offset beyond file size
    let result = mmap.access_archived_at::<ArchivedCacheEntry>(data.len() + 100);

    assert!(matches!(
        result,
        Err(MmapError::FileTooSmall {
            expected: _,
            actual: _
        })
    ));
}

#[test]
fn test_handle_access_archived_at_offset_too_large() {
    use crate::storage::ArchivedCacheEntry;

    let data = create_serialized_entry();
    let file = create_test_file(&data);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    // Offset beyond file size
    let result = handle.access_archived_at::<ArchivedCacheEntry>(data.len() + 100);

    assert!(matches!(
        result,
        Err(MmapError::FileTooSmall {
            expected: _,
            actual: _
        })
    ));
}

#[test]
fn test_shrink_readonly_fails() {
    let content = b"content to shrink";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    let result = mmap.shrink(5);
    assert!(matches!(result, Err(MmapError::ResizeFailed(_))));

    // Verify error message mentions read-only
    if let Err(MmapError::ResizeFailed(msg)) = result {
        assert!(msg.contains("read-only"));
    }
}

#[test]
fn test_shrink_to_larger_size_fails() {
    let content = b"short";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    // Try to "shrink" to a larger size
    let result = mmap.shrink(100);
    assert!(matches!(result, Err(MmapError::ResizeFailed(_))));

    // Verify error message mentions size comparison
    if let Err(MmapError::ResizeFailed(msg)) = result {
        assert!(msg.contains("smaller"));
    }
}

#[test]
fn test_grow_to_smaller_size_fails() {
    let content = b"longer content here for grow test";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    // Try to "grow" to a smaller size
    let result = mmap.grow(5);
    assert!(matches!(result, Err(MmapError::ResizeFailed(_))));

    // Verify error message mentions size comparison
    if let Err(MmapError::ResizeFailed(msg)) = result {
        assert!(msg.contains("larger"));
    }
}

#[test]
fn test_copy_on_write_flush() {
    let content = b"copy on write flush";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::copy_on_write()).expect("Failed to open mmap");

    // Modify the copy
    let slice = mmap.as_mut_slice().expect("Should be writable");
    slice[0] = b'C';

    // Flush operations should work
    mmap.flush().expect("flush failed");
    mmap.flush_async().expect("flush_async failed");
    mmap.flush_range(0, 1).expect("flush_range failed");

    // Original file should be unchanged
    let original = std::fs::read(file.path()).expect("Failed to read");
    assert_eq!(original[0], b'c');
}

#[test]
fn test_mmap_config_default() {
    let config = MmapConfig::default();

    assert_eq!(config.mode, MmapMode::ReadOnly);
    assert!(!config.populate);
    assert!(config.offset.is_none());
    assert!(config.len.is_none());
}

#[test]
fn test_mmap_mode_equality() {
    assert_eq!(MmapMode::ReadOnly, MmapMode::ReadOnly);
    assert_eq!(MmapMode::ReadWrite, MmapMode::ReadWrite);
    assert_eq!(MmapMode::CopyOnWrite, MmapMode::CopyOnWrite);
    assert_ne!(MmapMode::ReadOnly, MmapMode::ReadWrite);
    assert_ne!(MmapMode::ReadWrite, MmapMode::CopyOnWrite);
}

#[test]
fn test_mmap_mode_debug() {
    let debug_str = format!("{:?}", MmapMode::ReadOnly);
    assert!(debug_str.contains("ReadOnly"));

    let debug_str = format!("{:?}", MmapMode::ReadWrite);
    assert!(debug_str.contains("ReadWrite"));

    let debug_str = format!("{:?}", MmapMode::CopyOnWrite);
    assert!(debug_str.contains("CopyOnWrite"));
}

#[test]
fn test_mmap_config_clone() {
    let config = MmapConfig::read_write()
        .with_populate()
        .with_offset(1024)
        .with_len(4096);

    let cloned = config.clone();

    assert_eq!(config.mode, cloned.mode);
    assert_eq!(config.populate, cloned.populate);
    assert_eq!(config.offset, cloned.offset);
    assert_eq!(config.len, cloned.len);
}

#[test]
fn test_mmap_config_debug() {
    let config = MmapConfig::read_only();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("MmapConfig"));
    assert!(debug_str.contains("mode"));
}

#[test]
fn test_mmap_error_display() {
    let err = MmapError::EmptyFile;
    let msg = format!("{}", err);
    assert!(msg.contains("empty"));

    let err = MmapError::FileTooSmall {
        expected: 100,
        actual: 50,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("50"));
    assert!(msg.contains("100"));

    let err = MmapError::AlignmentError {
        offset: 7,
        alignment: 16,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("7"));
    assert!(msg.contains("16"));

    let err = MmapError::ResizeFailed("test error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("test error"));

    let err = MmapError::ValidationFailed("validation error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("validation"));
}

#[test]
fn test_mmap_error_debug() {
    let err = MmapError::EmptyFile;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("EmptyFile"));
}

#[test]
fn test_readonly_as_mut_slice_returns_none() {
    let content = b"readonly content";
    let file = create_test_file(content);

    let mut mmap =
        MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    assert!(mmap.as_mut_slice().is_none());
}

#[test]
fn test_mutable_as_slice_via_read_write() {
    let content = b"mutable content test";
    let file = create_test_file(content);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_write()).expect("Failed to open mmap");

    // as_slice should work on mutable mapping
    assert_eq!(mmap.as_slice(), content);
}

#[test]
fn test_mutable_as_slice_via_copy_on_write() {
    let content = b"copy on write content";
    let file = create_test_file(content);

    let mmap =
        MmapFile::open(file.path(), MmapConfig::copy_on_write()).expect("Failed to open mmap");

    // as_slice should work on copy-on-write mapping
    assert_eq!(mmap.as_slice(), content);
}

#[test]
fn test_handle_access_archived() {
    use crate::storage::ArchivedCacheEntry;

    let data = create_serialized_entry();
    let file = create_test_file(&data);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    let archived = handle
        .access_archived::<ArchivedCacheEntry>()
        .expect("Failed to access archived");

    assert_eq!(archived.tenant_id, 12345);
    assert_eq!(archived.context_hash, 67890);
}

#[test]
fn test_mmap_file_is_empty() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("nonempty.bin");

    let mmap = MmapFile::create(&path, 100, MmapConfig::default()).expect("Failed to create mmap");

    // A file with size 100 should not be empty
    assert!(!mmap.is_empty());
}

#[test]
fn test_handle_is_empty() {
    let content = b"nonempty content";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    // Non-empty content means is_empty returns false
    assert!(!handle.is_empty());
}

#[test]
fn test_access_archived_alignment_error() {
    // Create file with content long enough, but we'll access at unaligned offset
    let content: Vec<u8> = (0..128).collect();
    let file = create_test_file(&content);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    // Try to access at offset 1 (not aligned to RKYV_ALIGNMENT=16)
    let result = mmap.access_archived_at::<crate::storage::ArchivedCacheEntry>(1);

    assert!(matches!(
        result,
        Err(MmapError::AlignmentError {
            offset: 1,
            alignment: super::RKYV_ALIGNMENT
        })
    ));
}

#[test]
fn test_access_archived_validation_error() {
    // Create file with invalid rkyv data (random bytes)
    let content = b"this is definitely not a valid rkyv serialized CacheEntry structure";
    let file = create_test_file(content);

    let mmap = MmapFile::open(file.path(), MmapConfig::read_only()).expect("Failed to open mmap");

    // Access at offset 0 should fail validation
    let result = mmap.access_archived::<crate::storage::ArchivedCacheEntry>();

    assert!(matches!(result, Err(MmapError::ValidationFailed(_))));
}

#[test]
fn test_handle_access_archived_alignment_error() {
    // Create file with content long enough, but we'll access at unaligned offset
    let content: Vec<u8> = (0..128).collect();
    let file = create_test_file(&content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    // Try to access at offset 1 (not aligned to RKYV_ALIGNMENT=16)
    let result = handle.access_archived_at::<crate::storage::ArchivedCacheEntry>(1);

    assert!(matches!(
        result,
        Err(MmapError::AlignmentError {
            offset: 1,
            alignment: super::RKYV_ALIGNMENT
        })
    ));
}

#[test]
fn test_handle_access_archived_validation_error() {
    // Create file with invalid rkyv data
    let content = b"this is definitely not a valid rkyv serialized CacheEntry structure";
    let file = create_test_file(content);

    let handle = MmapFileHandle::open(file.path()).expect("Failed to open handle");

    // Access at offset 0 should fail validation
    let result = handle.access_archived::<crate::storage::ArchivedCacheEntry>();

    assert!(matches!(result, Err(MmapError::ValidationFailed(_))));
}

#[test]
fn test_config_with_offset() {
    // Create a file with enough content
    let content: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
    let file = create_test_file(&content);

    let config = MmapConfig::read_only().with_offset(4096);

    let mmap = MmapFile::open(file.path(), config).expect("Failed to open mmap");

    // The mapped region starts at offset 4096
    assert_eq!(mmap.as_slice()[0], (4096 % 256) as u8);
}

#[test]
fn test_config_with_len() {
    let content: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let file = create_test_file(&content);

    let config = MmapConfig::read_only().with_len(100);

    let mmap = MmapFile::open(file.path(), config).expect("Failed to open mmap");

    assert_eq!(mmap.len(), 100);
}

#[test]
fn test_config_with_populate() {
    let content = b"content to populate";
    let file = create_test_file(content);

    let config = MmapConfig::read_only().with_populate();

    let mmap = MmapFile::open(file.path(), config).expect("Failed to open mmap");

    assert_eq!(mmap.as_slice(), content);
}
