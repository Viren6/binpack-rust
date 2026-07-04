use std::io::{self};
use std::io::{Read, Seek};
use thiserror::Error;

use crate::common::{
    binpack_error::BinpackError, compressed_training_file_reader::CompressedTrainingDataFileReader,
    entry::PackedTrainingDataEntry, entry::TrainingDataEntry,
};

use super::move_score_list_reader::PackedMoveScoreListReader;

const SUGGESTED_CHUNK_SIZE: usize = 8192;

#[derive(Debug, Error)]
pub enum CompressedReaderError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
    #[error("End of file reached")]
    EndOfFile,
    #[error("Binpack error: {0}")]
    BinpackError(#[from] BinpackError),
}

type Result<T> = std::result::Result<T, CompressedReaderError>;

/// Read the next raw binpack chunk payload into `buffer`.
///
/// Returns `Ok(false)` when the stream is already at EOF. Otherwise this reads
/// the next chunk header and payload, resizes `buffer` to the chunk size, and
/// overwrites it with the chunk bytes before returning `Ok(true)`.
///
/// This helper does not keep any reader state beyond the current stream
/// position, so it can be called repeatedly on the same file handle as long as
/// the handle remains positioned at the start of the next chunk.
pub fn read_chunk_into<T: Read + Seek>(file: &mut T, buffer: &mut Vec<u8>) -> Result<bool> {
    let mut reader = CompressedTrainingDataFileReader::new(file)?;

    if !reader.has_next_chunk() {
        return Ok(false);
    }

    reader.read_next_chunk_into(buffer)?;

    Ok(true)
}

pub fn parse_chunk(chunk: &[u8]) -> Vec<TrainingDataEntry> {
    let mut reader = ChunkReader::default();
    let mut entries = Vec::new();

    while reader.has_next(chunk) {
        entries.push(reader.next(chunk));
    }

    entries
}

/// Reads Stockfish binpacks and returns a TrainingDataEntry
/// for each encoded entry.
#[derive(Debug)]
pub struct CompressedTrainingDataEntryReader<T: Read + Seek> {
    chunk: Vec<u8>,
    chunk_reader: ChunkReader,
    input_file: Option<CompressedTrainingDataFileReader<T>>,
    is_end: bool,
}

#[derive(Debug, Default)]
pub struct ChunkReader {
    movelist_reader: Option<PackedMoveScoreListReader>,
    offset: usize,
    is_end: bool,
}

/*
Search for EBNF: ..., to find the implementation.

File         = Block*
Block        = ChunkHeader Chain*
ChunkHeader  = Magic ChunkSize
Magic        = "BINP"
ChunkSize    = UINT32LE               (* 4 bytes, little endian *)

Chain        = Stem Count MoveText
Stem         = Position Move Score PlyResult Rule50 DrawScore
Count        = UINT16BE               (* 2 bytes, big endian *)
MoveText     = MoveScore*

(* Stem components - total 34 bytes *)
Position     = CompressedPosition     (* 24 bytes *)
Move         = CompressedMove         (* 2 bytes *)
Score        = INT16BE                (* 2 bytes, big endian, signed *)
PlyResult    = UINT8                  (* 2 byte, big endian unsigned *)
Rule50       = UINT16BE               (* 2 bytes, big endian *)
DrawScore    = INT16BE                (* 2 bytes, big endian, signed; 2nd value channel *)

(* MoveText components *)
MoveScore    = EncodedMove EncodedScore EncodedDraw

(* Encoded components *)
EncodedMove  = VARLEN_UINT            (* Variable length encoding *)
EncodedScore = VARLEN_INT             (* Variable length encoding; sign-flipped per ply *)
EncodedDraw  = VARLEN_INT             (* Variable length encoding; NOT sign-flipped (side-symmetric) *)
*/

// EBNF: File
impl<T: Read + Seek> CompressedTrainingDataEntryReader<T> {
    /// Create a new CompressedTrainingDataEntryReader,
    /// reading from the file at the given path.
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use sfbinpack::CompressedTrainingDataEntryReader;
    ///
    /// let file = File::options().read(true).write(false).create(false).open("in.binpack").unwrap();
    /// let mut reader = CompressedTrainingDataEntryReader::new(file).unwrap();
    ///
    /// while reader.has_next() {
    ///     let entry = reader.next();
    /// }
    /// ```
    pub fn new(file: T) -> Result<Self> {
        let chunk = Vec::with_capacity(SUGGESTED_CHUNK_SIZE);

        let mut reader = Self {
            chunk,
            chunk_reader: ChunkReader::default(),
            input_file: Some(CompressedTrainingDataFileReader::new(file)?),
            is_end: false,
        };

        if !reader.load_next_chunk()? {
            reader.is_end = true;
            return Err(CompressedReaderError::EndOfFile);
        }

        Ok(reader)
    }

    pub fn into_inner(&mut self) -> io::Result<T> {
        self.input_file.take().unwrap().into_inner()
    }

    /// Get how much of the file has been read so far
    pub fn read_bytes(&self) -> u64 {
        self.input_file.as_ref().unwrap().read_bytes()
    }

    /// Read the next raw binpack chunk payload into `buffer`.
    ///
    /// Returns `Ok(false)` when no more chunks are available. Otherwise this
    /// reads the next chunk header and payload, resizes `buffer` to the chunk
    /// size, and overwrites it with the chunk bytes before returning `Ok(true)`.
    pub fn read_next_chunk_into(&mut self, buffer: &mut Vec<u8>) -> Result<bool> {
        if !self.input_file.as_mut().unwrap().has_next_chunk() {
            return Ok(false);
        }

        self.input_file
            .as_mut()
            .unwrap()
            .read_next_chunk_into(buffer)?;

        Ok(true)
    }

    /// Parse all entries from a single chunk payload.
    pub fn parse_chunk(chunk: &[u8]) -> Vec<TrainingDataEntry> {
        parse_chunk(chunk)
    }

    /// Check if there are more TrainingDataEntry to read
    pub fn has_next(&self) -> bool {
        !self.is_end
    }

    /// Check if the next entry is a continuation of the last returned entry from next()
    pub fn is_next_entry_continuation(&self) -> bool {
        if let Some(ref reader) = self.chunk_reader.movelist_reader {
            return reader.has_next();
        }

        false
    }

    /// Get the next TrainingDataEntry
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> TrainingDataEntry {
        let entry = self.chunk_reader.next(&self.chunk);

        if !self.chunk_reader.has_next(&self.chunk) {
            self.fetch_next_chunk_if_needed();
        }

        entry
    }

    // EBNF: BLOCK
    fn fetch_next_chunk_if_needed(&mut self) {
        if self.chunk_reader.has_next(&self.chunk) {
            return;
        }

        if self.load_next_chunk().unwrap() {
            return;
        }

        self.is_end = true;
    }

    fn load_next_chunk(&mut self) -> Result<bool> {
        if !self.input_file.as_mut().unwrap().has_next_chunk() {
            return Ok(false);
        }

        self.input_file
            .as_mut()
            .unwrap()
            .read_next_chunk_into(&mut self.chunk)?;

        self.chunk_reader = ChunkReader::default();

        Ok(true)
    }
}

impl ChunkReader {
    /// Check whether another entry can be read from this chunk.
    pub fn has_next(&self, chunk: &[u8]) -> bool {
        if self
            .movelist_reader
            .as_ref()
            .is_some_and(|reader| reader.has_next())
        {
            return true;
        }

        !self.is_end && self.offset + PackedTrainingDataEntry::byte_size() + 2 <= chunk.len()
    }

    /// Read the next entry from this chunk.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self, chunk: &[u8]) -> TrainingDataEntry {
        if let Some(ref mut reader) = self.movelist_reader {
            let entry = reader.next_entry(&chunk[self.offset..]);

            if !reader.has_next() {
                self.offset += reader.num_read_bytes();
                self.movelist_reader = None;
                self.finish_if_at_end(chunk);
            }

            return entry;
        }

        // We don't have a movelist reader, so we first need to extract the "stem" information

        // EBNF: Stem
        let entry = self.read_entry(chunk);

        // EBNF: Count
        let num_plies = self.read_plies(chunk);

        if num_plies > 0 {
            // EBNF: MoveText
            self.movelist_reader = Some(PackedMoveScoreListReader::new(entry, num_plies));
        } else {
            self.finish_if_at_end(chunk);
        }

        entry
    }

    fn read_entry(&mut self, chunk: &[u8]) -> TrainingDataEntry {
        let size = PackedTrainingDataEntry::byte_size();

        debug_assert!(self.offset + size <= chunk.len());

        let packed = PackedTrainingDataEntry::from_slice(&chunk[self.offset..self.offset + size]);

        self.offset += size;

        packed.unpack_entry()
    }

    fn read_plies(&mut self, chunk: &[u8]) -> u16 {
        let ply = ((chunk[self.offset] as u16) << 8) | (chunk[self.offset + 1] as u16);
        self.offset += 2;
        ply
    }

    fn finish_if_at_end(&mut self, chunk: &[u8]) {
        if self.offset + PackedTrainingDataEntry::byte_size() + 2 > chunk.len() {
            self.is_end = true;
        }
    }
}

impl CompressedTrainingDataEntryReader<io::Cursor<Vec<u8>>> {
    /// Create a reader from an owned byte buffer.
    ///
    /// This is convenient for wasm environments where binpack data is often
    /// already available in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::new(io::Cursor::new(bytes))
    }
}

impl<'a> CompressedTrainingDataEntryReader<io::Cursor<&'a [u8]>> {
    /// Create a reader from a borrowed byte slice.
    pub fn from_slice(bytes: &'a [u8]) -> Result<Self> {
        Self::new(io::Cursor::new(bytes))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::chess::{
        coords::Square,
        piece::Piece,
        position::Position,
        r#move::{Move, MoveType},
    };

    use super::*;
    use crate::CompressedTrainingDataEntryWriter;

    /// A valid 3-ply continuation chain (stem + 2 continuations) with a varying
    /// draw channel, to exercise the no-flip draw delta on read-back.
    fn sample_chain() -> Vec<TrainingDataEntry> {
        vec![
            TrainingDataEntry {
                pos: Position::from_fen("1q5b/1r5k/4p2p/1b2P1pN/3p4/6PP/1nP3B1/1Q2B1K1 w - - 0 35")
                    .unwrap(),
                mv: Move::new(Square::new(10), Square::new(26), MoveType::Normal, Piece::none()),
                score: -201,
                ply: 68,
                result: 0,
                draw_score: 100,
            },
            TrainingDataEntry {
                pos: Position::from_fen("1q5b/1r5k/4p2p/1b2P1pN/2Pp4/6PP/1n4B1/1Q2B1K1 b - - 0 35")
                    .unwrap(),
                mv: Move::new(Square::new(27), Square::new(19), MoveType::Normal, Piece::none()),
                score: 254,
                ply: 69,
                result: 0,
                draw_score: -3000,
            },
            TrainingDataEntry {
                pos: Position::from_fen("1q5b/1r5k/4p2p/1b2P1pN/2P5/3p2PP/1n4B1/1Q2B1K1 w - - 0 36")
                    .unwrap(),
                mv: Move::new(Square::new(14), Square::new(49), MoveType::Normal, Piece::none()),
                score: -220,
                ply: 70,
                result: 0,
                draw_score: 12345,
            },
        ]
    }

    fn write_to_bytes(entries: &[TrainingDataEntry]) -> Vec<u8> {
        let mut writer = CompressedTrainingDataEntryWriter::new_in_memory().unwrap();
        for e in entries {
            writer.write_entry(e).unwrap();
        }
        writer.into_bytes().unwrap()
    }

    #[test]
    fn test_reader_roundtrip() {
        // Write a chain, read it back through the reader, and require every field
        // (score AND draw) to match. Exercises stem + continuation decoding.
        let entries = sample_chain();
        let bytes = write_to_bytes(&entries);

        let mut reader = CompressedTrainingDataEntryReader::from_bytes(bytes).unwrap();
        let mut got: Vec<TrainingDataEntry> = Vec::new();
        while reader.has_next() {
            got.push(reader.next());
        }

        assert_eq!(got, entries);
    }

    #[test]
    fn test_parse_chunk() {
        // The whole (small) chain lands in a single chunk. Strip the 8-byte
        // "BINP" + size header and parse the raw chunk payload directly.
        let bytes = write_to_bytes(&sample_chain());
        let entries = parse_chunk(&bytes[8..]);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[2].draw_score, 12345);
    }

    #[test]
    fn test_big_deltas_roundtrip() {
        // Extreme score/draw values near the i16 limits (full 4-block VLE deltas
        // with wrap-around) must survive the read-back exactly.
        let mut entries = sample_chain();
        entries[0].score = -31999;
        entries[0].draw_score = 30000;
        entries[1].score = -1500;
        entries[1].draw_score = -30000;
        entries[2].score = 32000;
        entries[2].draw_score = 30000;

        let bytes = write_to_bytes(&entries);
        let mut reader = CompressedTrainingDataEntryReader::from_bytes(bytes).unwrap();
        let mut got: Vec<TrainingDataEntry> = Vec::new();
        while reader.has_next() {
            got.push(reader.next());
        }
        assert_eq!(got, entries);
    }

    // Regression test for https://github.com/Disservin/binpack-rust/issues/17
    #[test]
    #[should_panic]
    fn test_reader_no_moves() {
        // Safe API UB: the reader builds a BitReader from a raw pointer without
        // tracking length. A crafted chunk with num_plies > 0 but no movetext
        // bytes triggers an out-of-bounds read. Build a valid (new-format) stem
        // via the writer's packer so this stays in sync with the layout.
        let stem = sample_chain()[0];
        let entry_bytes = PackedTrainingDataEntry::from_entry(&stem).data;

        // num_plies = 1, but movetext is empty.
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&entry_bytes);
        chunk.extend_from_slice(&1u16.to_be_bytes());

        // File header: "BINP" + chunk_size (LE).
        let mut file = Vec::new();
        file.extend_from_slice(b"BINP");
        file.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        file.extend_from_slice(&chunk);

        let cursor = Cursor::new(file);
        let mut reader = CompressedTrainingDataEntryReader::new(cursor).unwrap();

        // First next() returns the stem entry and sets movelist_reader.
        let _ = reader.next();
        // Second next() consumes movetext via BitReader and triggers OOB.
        let _ = reader.next();
    }
}
