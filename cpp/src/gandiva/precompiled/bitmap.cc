// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// BitMap functions

//#include "arrow/util/bit_util.h"

extern "C" {

#include "./types.h"
#include <stddef.h>
#include <string.h>

#define BITS_TO_BYTES(x) ((x + 7) / 8)
#define BITS_TO_WORDS(x) ((x + 63) / 64)

#define POS_TO_BYTE_INDEX(p) (p / 8)
#define POS_TO_BIT_INDEX(p) (p % 8)

//JJH: From arrow/util/bit_util.h:

// Bitmask selecting the k-th bit in a byte
static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

// the bitwise complement version of kBitmask
static constexpr uint8_t kFlippedBitmask[] = {254, 253, 251, 247, 239, 223, 191, 127};

// Bitmask selecting the (k - 1) preceding bits in a byte
static constexpr uint8_t kPrecedingBitmask[] = {0, 1, 3, 7, 15, 31, 63, 127};
static constexpr uint8_t kPrecedingWrappingBitmask[] = {255, 1, 3, 7, 15, 31, 63, 127};

// the bitwise complement version of kPrecedingBitmask
static constexpr uint8_t kTrailingBitmask[] = {255, 254, 252, 248, 240, 224, 192, 128};

static inline bool GetBit(const uint8_t* bits, uint64_t i) {
  return (bits[i >> 3] >> (i & 0x07)) & 1;
}

// Gets the i-th bit from a byte. Should only be used with i <= 7.
static inline bool GetBitFromByte(uint8_t byte, uint8_t i) { return byte & kBitmask[i]; }

static inline void ClearBit(uint8_t* bits, int64_t i) {
  bits[i / 8] &= kFlippedBitmask[i % 8];
}

static inline void SetBit(uint8_t* bits, int64_t i) { bits[i / 8] |= kBitmask[i % 8]; }

static inline void SetBitTo(uint8_t* bits, int64_t i, bool bit_is_set) {
  // https://graphics.stanford.edu/~seander/bithacks.html
  // "Conditionally set or clear bits without branching"
  // NOTE: this seems to confuse Valgrind as it reads from potentially
  // uninitialized memory
  bits[i / 8] ^= static_cast<uint8_t>(-static_cast<uint8_t>(bit_is_set) ^ bits[i / 8]) &
                 kBitmask[i % 8];
}

/// \brief set or clear a range of bits quickly
static inline void SetBitsTo(uint8_t* bits, int64_t start_offset, int64_t length,
                             bool bits_are_set) {
  if (length == 0) return;

  const auto i_begin = start_offset;
  const auto i_end = start_offset + length;
  const uint8_t fill_byte = static_cast<uint8_t>(-static_cast<uint8_t>(bits_are_set));

  const auto bytes_begin = i_begin / 8;
  const auto bytes_end = i_end / 8 + 1;

  const auto first_byte_mask = kPrecedingBitmask[i_begin % 8];
  const auto last_byte_mask = kTrailingBitmask[i_end % 8];

  if (bytes_end == bytes_begin + 1) {
    // set bits within a single byte
    const auto only_byte_mask =
        i_end % 8 == 0 ? first_byte_mask
                       : static_cast<uint8_t>(first_byte_mask | last_byte_mask);
    bits[bytes_begin] &= only_byte_mask;
    bits[bytes_begin] |= static_cast<uint8_t>(fill_byte & ~only_byte_mask);
    return;
  }

  // set/clear trailing bits of first byte
  bits[bytes_begin] &= first_byte_mask;
  bits[bytes_begin] |= static_cast<uint8_t>(fill_byte & ~first_byte_mask);

  if (bytes_end - bytes_begin > 2) {
    // set/clear whole bytes
    memset(bits + bytes_begin + 1, fill_byte,
                static_cast<size_t>(bytes_end - bytes_begin - 2));
  }

  if (i_end % 8 == 0) return;

  // set/clear leading bits of last byte
  bits[bytes_end - 1] &= last_byte_mask;
  bits[bytes_end - 1] |= static_cast<uint8_t>(fill_byte & ~last_byte_mask);
}

FORCE_INLINE
bool bitMapGetBit(const uint8_t* bmap, int64_t position) {
  return GetBit(bmap, position);
}

FORCE_INLINE
bool bitMapValidityGetBit(const uint8_t* bmap, int64_t position) {
  if (bmap == nullptr) {
    // if validity bitmap is null, all entries are valid.
    return true;
  } else {
    return bitMapGetBit(bmap, position);
  }
}

FORCE_INLINE
void bitMapSetBit(uint8_t* bmap, int64_t position, bool value) {
  SetBitTo(bmap, position, value);
}

// Clear the bit if value = false. Does nothing if value = true.
FORCE_INLINE
void bitMapClearBitIfFalse(uint8_t* bmap, int64_t position, bool value) {
  if (!value) {
    ClearBit(bmap, position);
  }
}

}  // extern "C"
