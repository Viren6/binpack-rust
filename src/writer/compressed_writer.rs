use std::io::Write;
use std::io::{self};
use thiserror::Error;

use crate::{
    chess::{position::Position, r#move::Move},
    common::{
        compressed_training_file_writer::CompressedTrainingDataFileWriter,
        entry::PackedTrainingDataEntry, entry::TrainingDataEntry,
    },
};

use super::move_score_list::PackedMoveScoreList;

const KI_B: usize = 1024;
const MI_B: usize = 1024 * KI_B;

const SUGGESTED_CHUNK_SIZE: usize = MI_B;
const MAX_MOVELIST_SIZE: usize = 10 * KI_B;

#[derive(Debug, Error)]
pub enum CompressedWriterError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
    #[error("End of file reached")]
    EndOfFile,
}

type Result<T> = std::result::Result<T, CompressedWriterError>;

/// Write Stockfish binpacks from TrainingDataEntry's
/// to a file.
#[derive(Debug)]
pub struct CompressedTrainingDataEntryWriter<T: Write> {
    output_file: Option<CompressedTrainingDataFileWriter<T>>,
    last_entry: TrainingDataEntry,
    movelist: PackedMoveScoreList,
    packed_size: usize,
    packed_entries: Vec<u8>,
    is_first: bool,
}

impl<T: Write> CompressedTrainingDataEntryWriter<T> {
    /// Create a new CompressedTrainingDataEntryWriter,
    /// writing to the file at the given path.
    /// The file will only be completely saved when the writer is dropped!
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use sfbinpack::CompressedTrainingDataEntryWriter;
    ///
    /// let file = File::options().read(true).write(true).create(true).open("out.binpack").unwrap();
    /// let mut writer = CompressedTrainingDataEntryWriter::new(file).unwrap();
    /// ```
    pub fn new(file: T) -> Result<Self> {
        let writer = Self {
            output_file: Some(CompressedTrainingDataFileWriter::new(file)?),
            last_entry: TrainingDataEntry {
                ply: 0xFFFF, // never a continuation
                result: 0x7FFF,
                pos: Position::default(),
                mv: Move::default(),
                score: 0,
                draw_score: 0,
            },
            movelist: PackedMoveScoreList::new(),
            packed_size: 0,
            packed_entries: vec![0u8; SUGGESTED_CHUNK_SIZE + MAX_MOVELIST_SIZE],
            is_first: true,
        };
        Ok(writer)
    }

    pub fn into_inner(&mut self) -> io::Result<T> {
        self.output_file.take().unwrap().into_inner()
    }

    pub fn written_bytes(&self) -> u64 {
        self.output_file.as_ref().unwrap().written_bytes()
    }

    /// Write a single entry to the file
    pub fn write_entry(&mut self, entry: &TrainingDataEntry) -> Result<()> {
        let is_cont = self.last_entry.is_continuation(entry);

        if is_cont {
            self.movelist
                .add_move_score(&entry.pos, entry.mv, entry.score, entry.draw_score);
        } else {
            if !self.is_first {
                self.write_movelist();
            }

            if self.packed_size >= SUGGESTED_CHUNK_SIZE {
                match self
                    .output_file
                    .as_mut()
                    .unwrap()
                    .append(&self.packed_entries[..self.packed_size])
                {
                    Ok(_) => {}
                    Err(e) => {
                        return Err(CompressedWriterError::Io(e));
                    }
                }
                self.packed_size = 0;
            }

            let packed = PackedTrainingDataEntry::from_entry(entry);
            let packed_bytes: [u8; size_of::<PackedTrainingDataEntry>()] = packed.data;

            self.packed_entries
                [self.packed_size..self.packed_size + PackedTrainingDataEntry::byte_size()]
                .copy_from_slice(&packed_bytes);

            self.packed_size += PackedTrainingDataEntry::byte_size();

            self.movelist.clear(entry);
            self.is_first = false;
        }

        self.last_entry = *entry;
        Ok(())
    }

    pub fn flush_and_end(&mut self) {
        let _ = self.flush_packed();
    }

    pub fn flush(&mut self) {
        if let Some(file) = self.output_file.as_mut() {
            let _ = file.flush();
        }
    }

    /// Flush the buffer to the file, automatically called when the writer is dropped
    fn flush_packed(&mut self) -> Result<()> {
        if self.packed_size > 0 {
            if !self.is_first {
                self.write_movelist();
            }

            match self
                .output_file
                .as_mut()
                .unwrap()
                .append(&self.packed_entries[..self.packed_size])
            {
                Ok(_) => {}
                Err(e) => {
                    return Err(CompressedWriterError::Io(e));
                }
            }
            self.packed_size = 0;
        }

        if let Some(file) = self.output_file.as_mut() {
            file.flush()?;
        }

        Ok(())
    }

    fn write_movelist(&mut self) {
        self.packed_entries[self.packed_size] = (self.movelist.num_plies >> 8) as u8;
        self.packed_entries[self.packed_size + 1] = self.movelist.num_plies as u8;
        self.packed_size += 2;

        if self.movelist.num_plies > 0 {
            let movetext = self.movelist.movetext();
            self.packed_entries[self.packed_size..self.packed_size + movetext.len()]
                .copy_from_slice(movetext);
            self.packed_size += movetext.len();
        }
    }
}

impl CompressedTrainingDataEntryWriter<io::Cursor<Vec<u8>>> {
    /// Create an in-memory writer.
    ///
    /// This is convenient for wasm environments where the encoded binpack is
    /// usually returned as a byte buffer instead of being written to a file.
    pub fn new_in_memory() -> Result<Self> {
        Self::new(io::Cursor::new(Vec::new()))
    }

    /// Flush pending data and return the encoded binpack bytes.
    pub fn into_bytes(&mut self) -> Result<Vec<u8>> {
        self.flush_packed()?;
        let cursor = self.into_inner()?;
        Ok(cursor.into_inner())
    }
}

impl<T: Write> Drop for CompressedTrainingDataEntryWriter<T> {
    fn drop(&mut self) {
        if let Err(e) = self.flush_packed() {
            eprintln!("Error flushing writer: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{
        coords::Square,
        piece::Piece,
        position::Position,
        r#move::{Move, MoveType},
    };
    use crate::CompressedTrainingDataEntryReader;

    /// A valid 3-ply continuation chain (stem + 2 continuations), annotated with
    /// a *varying* draw-score channel so round-trips exercise the no-flip draw
    /// delta across continuations.
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

    fn read_all(bytes: Vec<u8>) -> Vec<TrainingDataEntry> {
        let mut reader = CompressedTrainingDataEntryReader::from_bytes(bytes).unwrap();
        let mut out = Vec::new();
        while reader.has_next() {
            out.push(reader.next());
        }
        out
    }

    #[test]
    fn test_write_read_roundtrip_with_draw() {
        // 3-ply chain -> 1 stem + 2 continuations; every field must round-trip
        // bit-exact, including the draw channel delta-coded without a sign flip.
        let entries = sample_chain();
        let round = read_all(write_to_bytes(&entries));
        assert_eq!(round, entries);
    }

    #[test]
    fn test_write_read_roundtrip_big_deltas() {
        // Extreme score and draw values (near the i16 limits, forcing full
        // 4-block VLE deltas and wrap-around) must still round-trip losslessly.
        let mut entries = sample_chain();
        entries[0].score = -31999;
        entries[0].draw_score = 30000;
        entries[1].score = -1500;
        entries[1].draw_score = -30000;
        entries[2].score = 32000;
        entries[2].draw_score = 30000;
        let round = read_all(write_to_bytes(&entries));
        assert_eq!(round, entries);
    }

    #[test]
    fn test_stem_only_roundtrip_with_draw() {
        // A lone stem entry (no continuations) carries its draw score in the
        // 34-byte packed entry.
        let entries = vec![sample_chain()[0]];
        let round = read_all(write_to_bytes(&entries));
        assert_eq!(round, entries);
        assert_eq!(round[0].draw_score, 100);
    }

    #[test]
    fn test_writer_into_bytes() {
        let bytes = write_to_bytes(&[sample_chain()[0]]);
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[..4], b"BINP");
    }
}
