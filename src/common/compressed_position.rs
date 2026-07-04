use crate::chess::{
    bitboard::Bitboard,
    castling_rights::CastlingRights,
    color::Color,
    coords::{FlatSquareOffset, Rank, Square},
    piece::Piece,
    piecetype::PieceType,
    position::Position,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressedPosition {
    occupied: Bitboard,
    packed_state: [u8; 16],
}

impl CompressedPosition {
    pub fn byte_size() -> usize {
        std::mem::size_of::<CompressedPosition>()
    }

    pub fn read_from_big_endian(data: &[u8]) -> Self {
        debug_assert!(data.len() >= 24);

        let occupied = Bitboard::new(
            ((data[0] as u64) << 56)
                | ((data[1] as u64) << 48)
                | ((data[2] as u64) << 40)
                | ((data[3] as u64) << 32)
                | ((data[4] as u64) << 24)
                | ((data[5] as u64) << 16)
                | ((data[6] as u64) << 8)
                | (data[7] as u64),
        );

        let mut packed_state = [0u8; 16];
        packed_state.copy_from_slice(&data[8..24]);

        Self {
            occupied,
            packed_state,
        }
    }

    pub fn write_to_big_endian(&self, data: &mut [u8]) {
        let occupied = self.occupied.bits();
        data[0] = (occupied >> 56) as u8;
        data[1] = ((occupied >> 48) & 0xFF) as u8;
        data[2] = ((occupied >> 40) & 0xFF) as u8;
        data[3] = ((occupied >> 32) & 0xFF) as u8;
        data[4] = ((occupied >> 24) & 0xFF) as u8;
        data[5] = ((occupied >> 16) & 0xFF) as u8;
        data[6] = ((occupied >> 8) & 0xFF) as u8;
        data[7] = (occupied & 0xFF) as u8;
        data[8..24].copy_from_slice(&self.packed_state[..16]);
    }

    pub fn decompress(&self) -> Position {
        let mut pos = Position::empty();
        pos.set_castling_rights(CastlingRights::NONE);

        // Castling rooks are marked with nibbles 13 (white) / 14 (black), but the
        // side (king-/queen-side) is NOT stored: it is inferred from the rook's
        // file relative to its king. That needs both kings placed first, so we
        // just collect the marked rook squares here and resolve rights after the
        // pass (see resolve_castling). DFRC/Chess960-general: no a/h assumption.
        let mut white_castle_rooks: u64 = 0;
        let mut black_castle_rooks: u64 = 0;

        let mut decompress_piece = |sq: Square, nibble: u8| {
            match nibble {
                0..=11 => {
                    pos.place(Piece::from_id(nibble as i32), sq);
                }
                12 => {
                    let rank = sq.rank();
                    if rank == Rank::FOURTH {
                        pos.place(Piece::WHITE_PAWN, sq);
                        pos.set_ep_square_unchecked(sq + FlatSquareOffset::new(0, -1));
                    } else {
                        // rank == Rank::FIFTH
                        pos.place(Piece::BLACK_PAWN, sq);
                        pos.set_ep_square_unchecked(sq + FlatSquareOffset::new(0, 1));
                    }
                }
                13 => {
                    pos.place(Piece::WHITE_ROOK, sq);
                    white_castle_rooks |= 1u64 << sq.index();
                }
                14 => {
                    pos.place(Piece::BLACK_ROOK, sq);
                    black_castle_rooks |= 1u64 << sq.index();
                }
                15 => {
                    pos.place(Piece::BLACK_KING, sq);
                    pos.set_side_to_move(Color::Black);
                }
                _ => unreachable!(),
            }
        };

        let mut squares_iter = self.occupied.iter();
        for chunk in self.packed_state.iter() {
            if let Some(sq) = squares_iter.next() {
                decompress_piece(sq, chunk & 0xF);
            } else {
                break;
            }

            if let Some(sq) = squares_iter.next() {
                decompress_piece(sq, chunk >> 4);
            } else {
                break;
            }
        }

        resolve_castling(&mut pos, Color::White, white_castle_rooks);
        resolve_castling(&mut pos, Color::Black, black_castle_rooks);

        pos
    }

    pub fn compress(pos: &Position) -> Self {
        let mut compressed = CompressedPosition {
            occupied: pos.occupied(),
            packed_state: [0u8; 16],
        };

        let rights = pos.castling_rights();
        let white_king = pos.king_sq(Color::White).index();
        let black_king = pos.king_sq(Color::Black).index();

        let pack_piece = |sq: Square| -> u8 {
            let piece = pos.piece_at(sq);
            let piece_id = piece.id();

            // Special case: pawn with en passant
            if piece.piece_type() == PieceType::Pawn {
                let ep_sq = pos.ep_square();
                if ep_sq != Square::NONE
                    && ((piece.color() == Color::White
                        && sq.rank() == Rank::FOURTH
                        && ep_sq == sq + FlatSquareOffset::new(0, -1))
                        || (piece.color() == Color::Black
                            && sq.rank() == Rank::FIFTH
                            && ep_sq == sq + FlatSquareOffset::new(0, 1)))
                {
                    return 12;
                }
            }

            // Special case: a castling rook (DFRC/Chess960-general). The rook is a
            // castling rook if it sits on its king's rank and the matching right is
            // held; queen-side = left of the king, king-side = right of it. The
            // side is inferred from the king's file, NOT hard-coded to a1/h1, so
            // rooks on any file are handled correctly.
            if piece == Piece::WHITE_ROOK && (sq.index() >> 3) == (white_king >> 3) {
                if (sq.index() & 7) < (white_king & 7) {
                    if rights.contains(CastlingRights::WHITE_QUEEN_SIDE) {
                        return 13;
                    }
                } else if rights.contains(CastlingRights::WHITE_KING_SIDE) {
                    return 13;
                }
            }
            if piece == Piece::BLACK_ROOK && (sq.index() >> 3) == (black_king >> 3) {
                if (sq.index() & 7) < (black_king & 7) {
                    if rights.contains(CastlingRights::BLACK_QUEEN_SIDE) {
                        return 14;
                    }
                } else if rights.contains(CastlingRights::BLACK_KING_SIDE) {
                    return 14;
                }
            }

            // Special case: black king when black to move
            if piece == Piece::BLACK_KING && pos.side_to_move() == Color::Black {
                return 15;
            }

            piece_id
        };

        let mut idx = 0;
        for (nibble_idx, sq) in compressed.occupied.iter().enumerate() {
            let nibble = pack_piece(sq);
            if nibble_idx % 2 == 0 {
                compressed.packed_state[idx] = nibble;
            } else {
                compressed.packed_state[idx] |= nibble << 4;
                idx += 1;
            }
        }

        compressed
    }
}

/// Set `color`'s castling rights from a bitboard of its marked castling-rook
/// squares. The stored format does not record king-/queen-side, so we infer it
/// from each rook's file relative to the king: a rook left of the king is the
/// queen-side rook, one to the right is the king-side rook. This is
/// DFRC/Chess960-general and makes no assumption about a/h-file rooks.
fn resolve_castling(pos: &mut Position, color: Color, mut rooks: u64) {
    if rooks == 0 {
        return;
    }

    let king_file = pos.king_sq(color).index() & 7;
    let (queen_side, king_side) = match color {
        Color::White => (
            CastlingRights::WHITE_QUEEN_SIDE,
            CastlingRights::WHITE_KING_SIDE,
        ),
        Color::Black => (
            CastlingRights::BLACK_QUEEN_SIDE,
            CastlingRights::BLACK_KING_SIDE,
        ),
    };

    while rooks != 0 {
        let file = rooks.trailing_zeros() & 7;
        rooks &= rooks - 1;
        if file < king_file {
            pos.add_castling_rights(queen_side);
        } else {
            pos.add_castling_rights(king_side);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_big_endian() {
        let data = [
            98, 121, 192, 21, 24, 76, 241, 100, 100, 106, 0, 4, 8, 48, 2, 17, 17, 145, 19, 117,
            247, 0, 0, 0,
        ];

        let compressed_pos = CompressedPosition::read_from_big_endian(&data);

        assert_eq!(
            CompressedPosition {
                occupied: Bitboard::new(7095913884733469028),
                packed_state: [100, 106, 0, 4, 8, 48, 2, 17, 17, 145, 19, 117, 247, 0, 0, 0]
            },
            compressed_pos
        );
    }

    #[test]
    fn test_compressed_position() {
        let data = [
            98, 121, 192, 21, 24, 76, 241, 100, 100, 106, 0, 4, 8, 48, 2, 17, 17, 145, 19, 117,
            247, 0, 0, 0,
        ];

        let compressed_pos = CompressedPosition::read_from_big_endian(&data);
        let pos = compressed_pos.decompress();

        assert_eq!(
            pos.fen().unwrap(),
            "1r3rk1/p2qnpb1/6pp/P1p1p3/3nN3/2QP2P1/R3PPBP/2B2RK1 b - - 0 1"
        );
    }

    #[test]
    #[should_panic(expected = "24")]
    fn test_too_small_data() {
        let data = [0; 23];

        let _ = CompressedPosition::read_from_big_endian(&data).decompress();
    }

    #[test]
    fn test_write_big_endian() {
        let data = [
            98, 121, 192, 21, 24, 76, 241, 100, 100, 106, 0, 4, 8, 48, 2, 17, 17, 145, 19, 117,
            247, 0, 0, 0,
        ];

        let compressed_pos = CompressedPosition::read_from_big_endian(&data);
        let mut new_data = [0; 24];
        compressed_pos.write_to_big_endian(&mut new_data);

        assert_eq!(data, new_data);
    }

    #[test]
    fn test_compress_decompress() {
        let pos =
            Position::from_fen("1r3rk1/p2qnpb1/6pp/P1p1p3/3nN3/2QP2P1/R3PPBP/2B2RK1 b - - 0 1")
                .unwrap();

        let compressed_pos = CompressedPosition::compress(&pos);
        let decompressed_pos = compressed_pos.decompress();

        assert_eq!(pos, decompressed_pos);
    }

    #[test]
    fn test_compress_decompress_2() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let compressed_pos = CompressedPosition::compress(&pos);
        let decompressed_pos = compressed_pos.decompress();

        assert_eq!(pos, decompressed_pos);
    }

    #[test]
    fn test_compress_decompress_3() {
        let pos = Position::from_fen("2r3k1/4bpp1/2Q1p2P/p3P3/1p6/4B1P1/P1r2PK1/3R1R2 b - - 0 30")
            .unwrap();

        let compressed_pos = CompressedPosition::compress(&pos);
        let decompressed_pos = compressed_pos.decompress();

        let position_without_fmt =
            Position::from_fen("2r3k1/4bpp1/2Q1p2P/p3P3/1p6/4B1P1/P1r2PK1/3R1R2 b - - 0 1")
                .unwrap();

        assert_eq!(position_without_fmt, decompressed_pos);
    }

    #[test]
    fn test_compress_decompress_dfrc_castling() {
        // Chess960: kings on the d-file with castling rooks on the b- and f-files
        // (NOT a/h). The old a1/h1-hard-coded encoding silently dropped these
        // rights; the king-file inference must round-trip all of KQkq.
        let pos = Position::from_fen("1r1k1r2/8/8/8/8/8/8/1R1K1R2 w KQkq - 0 1").unwrap();
        assert_eq!(pos.castling_rights(), CastlingRights::ALL);

        let compressed = CompressedPosition::compress(&pos);
        let decompressed = compressed.decompress();

        assert_eq!(decompressed.castling_rights(), CastlingRights::ALL);
        assert_eq!(pos, decompressed);
    }

    #[test]
    fn test_compress_decompress_dfrc_partial_rights() {
        // Only one side retains a right, and the rook is off the a/h files: white
        // keeps queen-side (rook c1, king e1), black keeps king-side (rook g8,
        // king b8). Must round-trip exactly, dropping nothing and adding nothing.
        // (CompressedPosition does not store the move clocks, so use 0 1.)
        let pos = Position::from_fen("1k4r1/8/8/8/8/8/8/2R1K3 w Qk - 0 1").unwrap();
        let expected = CastlingRights::WHITE_QUEEN_SIDE;
        assert!(pos.castling_rights().contains(expected));
        assert!(pos
            .castling_rights()
            .contains(CastlingRights::BLACK_KING_SIDE));

        let decompressed = CompressedPosition::compress(&pos).decompress();
        assert_eq!(pos, decompressed);
        assert_eq!(pos.castling_rights(), decompressed.castling_rights());
    }
}
