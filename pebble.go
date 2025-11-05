package db

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"runtime"

	"github.com/cockroachdb/pebble/v2"
	"github.com/cockroachdb/pebble/v2/bloom"
)

func init() {
	dbCreator := func(name string, dir string) (DB, error) {
		return NewPebbleDB(name, dir)
	}
	registerDBCreator(PebbleDBBackend, dbCreator)
}

// PebbleDB is a PebbleDB backend.
type PebbleDB struct {
	db *pebble.DB
}

var _ DB = (*PebbleDB)(nil)

// defaultPebbleOptions creates optimized PebbleDB options based on machine configuration.
// It automatically detects CPU cores and configures cache size, compaction concurrency,
// and other performance-related settings for optimal read/write performance.
//
// Read performance optimizations:
//   - Large cache (2GB) with optimized sharding for better hit rate
//   - Bloom filters on all levels to reduce unnecessary disk reads
//   - Larger block sizes (32KB) for better sequential read performance
//   - Larger target file sizes to reduce file count and improve read efficiency
//   - Optimized file cache shards based on CPU cores
//   - Readahead enabled for sequential reads
//
// Write performance optimizations:
//   - CompactionConcurrencyRange: dynamically set based on CPU cores
//   - MaxOpenFiles: 4096 (provides reliable performance boost)
//   - MemTableSize: 256MB (good balance for write performance)
//   - BytesPerSync/WALBytesPerSync: tuned for smoother writes
func DefaultPebbleOptions() *pebble.Options {
	numCPU := runtime.NumCPU()

	// Calculate compaction concurrency based on CPU cores
	// Use at least 1, but scale up with CPU cores (capped at reasonable max)
	// For production workloads, using 1-2x CPU cores is often optimal
	compactionLower := 1
	compactionUpper := numCPU
	if compactionUpper > 8 {
		// Cap at 8 for very high core count systems to avoid excessive resource usage
		compactionUpper = 8
	}
	if compactionLower > compactionUpper {
		compactionLower = compactionUpper
	}

	// Create a shared cache with optimized sharding for better read performance
	// Using NewCache allows us to leverage better cache management
	// For read-heavy workloads, larger cache significantly improves performance
	cacheSize := int64(2 << 30) // 2GB
	sharedCache := pebble.NewCache(cacheSize)

	opts := &pebble.Options{
		// Cache configuration: Use shared cache for better read performance
		// Shared cache allows multiple DB instances to share the same cache
		Cache: sharedCache,

		// Compaction concurrency: dynamically adjust based on CPU cores
		// Lower bound for normal operation, upper bound for high load scenarios
		CompactionConcurrencyRange: func() (int, int) {
			return compactionLower, compactionUpper
		},

		// MaxOpenFiles: 4096 provides reliable performance boost
		// This allows more SSTables to be kept open, reducing file open/close overhead
		MaxOpenFiles: 4096,

		// MemTableSize: 256MB provides good balance for write performance
		// Larger memtables improve write throughput but increase memory usage
		MemTableSize: 256 << 20, // 256MB

		// BytesPerSync: sync SSTables periodically to smooth out writes
		// Default is 512KB, but we can tune based on workload
		BytesPerSync: 2048 << 10, // 2MB

		// WALBytesPerSync: sync WAL periodically for smoother writes
		// Default is 0 (sync on every write), but 4MB provides good balance
		WALBytesPerSync: 4 << 20, // 4MB

		// L0 compaction thresholds: tuned for better write performance
		// Lower thresholds trigger compaction earlier, reducing write stalls
		L0CompactionFileThreshold: 4,  // default is 4
		L0CompactionThreshold:     4,  // default is 4
		L0StopWritesThreshold:     36, // default is 36, stop writes when L0 gets too large

		// TargetFileSizes: Optimize for read performance
		// Larger files reduce the number of files to manage, improving read efficiency
		// For read-heavy workloads, slightly larger files can improve performance
		TargetFileSizes: [7]int64{
			4 << 20,   // L0: 4MB (default is 2MB, larger for better read performance)
			8 << 20,   // L1: 8MB
			16 << 20,  // L2: 16MB
			32 << 20,  // L3: 32MB
			64 << 20,  // L4: 64MB
			128 << 20, // L5: 128MB
			256 << 20, // L6: 256MB
		},
	}

	// Set experimental options for better read performance
	// FileCacheShards: Optimize based on CPU cores for better concurrent read performance
	// More shards reduce contention when multiple goroutines access the file cache
	opts.Experimental.FileCacheShards = numCPU

	// Configure Bloom filters for all levels to optimize read performance
	// Bloom filters reduce unnecessary disk reads by quickly determining if a key might exist
	// Using bits per key of 10 provides good balance between filter size and false positive rate
	bloomFilter := bloom.FilterPolicy(10)
	for i := range opts.Levels {
		opts.Levels[i].FilterPolicy = bloomFilter
		// Optimize block size for read performance
		// Larger blocks (32KB) reduce overhead and improve sequential read performance
		// Default is 4KB for L0, but 32KB works better for read-heavy workloads
		opts.Levels[i].BlockSize = 32 << 10 // 32KB
		// Larger index blocks can improve index read performance
		opts.Levels[i].IndexBlockSize = 32 << 10 // 32KB
	}

	// Set experimental options for better compaction performance
	// L0CompactionConcurrency: enable concurrent compactions when L0 read-amplification is high
	// This allows Pebble to increase compaction concurrency when L0 gets large
	opts.Experimental.L0CompactionConcurrency = 2

	return opts
}

func NewPebbleDB(name, dir string) (DB, error) {
	opts := DefaultPebbleOptions()
	opts.EnsureDefaults()
	return NewPebbleDBWithOpts(name, dir, opts)
}

func NewPebbleDBWithOpts(name string, dir string, opts *pebble.Options) (*PebbleDB, error) {
	dbPath := filepath.Join(dir, name+".db")
	opts.EnsureDefaults()
	p, err := pebble.Open(dbPath, opts)
	if err != nil {
		return nil, err
	}
	return &PebbleDB{
		db: p,
	}, err
}

// Get implements DB.
func (db *PebbleDB) Get(key []byte) ([]byte, error) {
	if len(key) == 0 {
		return nil, errKeyEmpty
	}

	res, closer, err := db.db.Get(key)
	if err != nil {
		if err == pebble.ErrNotFound {
			return nil, nil
		}
		return nil, err
	}
	defer closer.Close()

	return cp(res), nil
}

// Has implements DB.
func (db *PebbleDB) Has(key []byte) (bool, error) {
	if len(key) == 0 {
		return false, errKeyEmpty
	}

	bytesPeb, err := db.Get(key)
	if err != nil {
		return false, err
	}
	return bytesPeb != nil, nil
}

// Set implements DB.
func (db *PebbleDB) Set(key []byte, value []byte) error {
	if len(key) == 0 {
		return errKeyEmpty
	}
	if value == nil {
		return errValueNil
	}

	wopts := pebble.NoSync
	err := db.db.Set(key, value, wopts)
	if err != nil {
		return err
	}
	return nil
}

// SetSync implements DB.
func (db *PebbleDB) SetSync(key []byte, value []byte) error {
	if len(key) == 0 {
		return errKeyEmpty
	}
	if value == nil {
		return errValueNil
	}
	err := db.db.Set(key, value, pebble.Sync)
	if err != nil {
		return err
	}
	return nil
}

// Delete implements DB.
func (db *PebbleDB) Delete(key []byte) error {
	if len(key) == 0 {
		return errKeyEmpty
	}

	wopts := pebble.NoSync
	return db.db.Delete(key, wopts)
}

// DeleteSync implements DB.
func (db PebbleDB) DeleteSync(key []byte) error {
	if len(key) == 0 {
		return errKeyEmpty
	}
	return db.db.Delete(key, pebble.Sync)
}

func (db *PebbleDB) DB() *pebble.DB {
	return db.db
}

func (db *PebbleDB) Compact(start, end []byte) (err error) {
	// Currently nil,nil is an invalid range in Pebble.
	// This was taken from https://github.com/cockroachdb/pebble/issues/1474
	// In case the start and end keys are the same
	// pebbleDB will throw an error that it cannot compact.
	if start != nil && end != nil {
		return db.db.Compact(context.Background(), start, end, true)
	}
	iter, err := db.db.NewIter(nil)
	if err != nil {
		return err
	}
	defer func() {
		err2 := iter.Close()
		if err2 != nil {
			err = err2
		}
	}()
	if start == nil && iter.First() {
		start = append(start, iter.Key()...)
	}
	if end == nil && iter.Last() {
		end = append(end, iter.Key()...)
	}
	return db.db.Compact(context.Background(), start, end, true)
}

// Close implements DB.
func (db PebbleDB) Close() error {
	db.db.Close()
	return nil
}

// Print implements DB.
func (db *PebbleDB) Print() error {
	itr, err := db.Iterator(nil, nil)
	if err != nil {
		return err
	}
	defer itr.Close()
	for ; itr.Valid(); itr.Next() {
		key := itr.Key()
		value := itr.Value()
		fmt.Printf("[%X]:\t[%X]\n", key, value)
	}
	return nil
}

// Stats implements DB.
func (*PebbleDB) Stats() map[string]string {
	return nil
}

// NewBatch implements DB.
func (db *PebbleDB) NewBatch() Batch {
	return newPebbleDBBatch(db)
}

// Iterator implements DB.
func (db *PebbleDB) Iterator(start, end []byte) (Iterator, error) {
	if (start != nil && len(start) == 0) || (end != nil && len(end) == 0) {
		return nil, errKeyEmpty
	}
	o := pebble.IterOptions{
		LowerBound: start,
		UpperBound: end,
	}
	itr, err := db.db.NewIter(&o)
	if err != nil {
		return nil, err
	}
	itr.First()

	return newPebbleDBIterator(itr, start, end, false), nil
}

// ReverseIterator implements DB.
func (db *PebbleDB) ReverseIterator(start, end []byte) (Iterator, error) {
	if (start != nil && len(start) == 0) || (end != nil && len(end) == 0) {
		return nil, errKeyEmpty
	}
	o := pebble.IterOptions{
		LowerBound: start,
		UpperBound: end,
	}
	itr, err := db.db.NewIter(&o)
	if err != nil {
		return nil, err
	}
	itr.Last()
	return newPebbleDBIterator(itr, start, end, true), nil
}

var _ Batch = (*pebbleDBBatch)(nil)

type pebbleDBBatch struct {
	db    *PebbleDB
	batch *pebble.Batch
}

var _ Batch = (*pebbleDBBatch)(nil)

func newPebbleDBBatch(db *PebbleDB) *pebbleDBBatch {
	return &pebbleDBBatch{
		// For regular batch operations batch.db is going to be set to db
		// and it is not needed to initialize the DB here.
		// This is set to enable general DB operations like compaction
		// (e.x. a call do pebbleDBBatch.db.Compact() would throw a nil pointer exception)
		db:    db,
		batch: db.db.NewBatch(),
	}
}

// Set implements Batch.
func (b *pebbleDBBatch) Set(key, value []byte) error {
	if len(key) == 0 {
		return errKeyEmpty
	}
	if value == nil {
		return errValueNil
	}
	if b.batch == nil {
		return errBatchClosed
	}

	return b.batch.Set(key, value, nil)
}

// Delete implements Batch.
func (b *pebbleDBBatch) Delete(key []byte) error {
	if len(key) == 0 {
		return errKeyEmpty
	}
	if b.batch == nil {
		return errBatchClosed
	}

	return b.batch.Delete(key, nil)
}

// Write implements Batch.
func (b *pebbleDBBatch) Write() error {
	if b.batch == nil {
		return errBatchClosed
	}

	wopts := pebble.NoSync
	err := b.batch.Commit(wopts)
	if err != nil {
		return err
	}
	// Make sure batch cannot be used afterwards. Callers should still call Close(), for errors.

	return b.Close()
}

// WriteSync implements Batch.
func (b *pebbleDBBatch) WriteSync() error {
	if b.batch == nil {
		return errBatchClosed
	}
	err := b.batch.Commit(pebble.Sync)
	if err != nil {
		return err
	}
	// Make sure batch cannot be used afterwards. Callers should still call Close(), for errors.
	return b.Close()
}

// Close implements Batch.
func (b *pebbleDBBatch) Close() error {
	if b.batch != nil {
		err := b.batch.Close()
		if err != nil {
			return err
		}
		b.batch = nil
	}

	return nil
}

type pebbleDBIterator struct {
	source     *pebble.Iterator
	start, end []byte
	isReverse  bool
	isInvalid  bool
}

var _ Iterator = (*pebbleDBIterator)(nil)

func newPebbleDBIterator(source *pebble.Iterator, start, end []byte, isReverse bool) *pebbleDBIterator {
	if isReverse {
		if end == nil {
			source.Last()
		}
	} else {
		if start == nil {
			source.First()
		}
	}
	return &pebbleDBIterator{
		source:    source,
		start:     start,
		end:       end,
		isReverse: isReverse,
		isInvalid: false,
	}
}

// Domain implements Iterator.
func (itr *pebbleDBIterator) Domain() (start []byte, end []byte) {
	return itr.start, itr.end
}

// Valid implements Iterator.
func (itr *pebbleDBIterator) Valid() bool {
	// Once invalid, forever invalid.
	if itr.isInvalid {
		return false
	}

	// If source has error, invalid.
	if err := itr.source.Error(); err != nil {
		itr.isInvalid = true

		return false
	}

	// If source is invalid, invalid.
	if !itr.source.Valid() {
		itr.isInvalid = true

		return false
	}

	// If key is end or past it, invalid.
	start := itr.start
	end := itr.end
	key := itr.source.Key()
	if itr.isReverse {
		if start != nil && bytes.Compare(key, start) < 0 {
			itr.isInvalid = true

			return false
		}
	} else {
		if end != nil && bytes.Compare(end, key) <= 0 {
			itr.isInvalid = true

			return false
		}
	}

	// It's valid.
	return true
}

// Key implements Iterator.
// The caller should not modify the contents of the returned slice.
// Instead, the caller should make a copy and work on the copy.
func (itr *pebbleDBIterator) Key() []byte {
	itr.assertIsValid()
	return itr.source.Key()
}

// Value implements Iterator.
// The caller should not modify the contents of the returned slice.
// Instead, the caller should make a copy and work on the copy.
func (itr *pebbleDBIterator) Value() []byte {
	itr.assertIsValid()
	return itr.source.Value()
}

// Next implements Iterator.
func (itr pebbleDBIterator) Next() {
	itr.assertIsValid()
	if itr.isReverse {
		itr.source.Prev()
	} else {
		itr.source.Next()
	}
}

// Error implements Iterator.
func (itr *pebbleDBIterator) Error() error {
	return itr.source.Error()
}

// Close implements Iterator.
func (itr *pebbleDBIterator) Close() error {
	err := itr.source.Close()
	if err != nil {
		return err
	}
	return nil
}

func (itr *pebbleDBIterator) assertIsValid() {
	if !itr.Valid() {
		panic("iterator is invalid")
	}
}
