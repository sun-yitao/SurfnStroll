#include "test.h"

#include <string.h>

typedef struct webpush_aes128gcm_decrypt_ok_test_s {
  const char* desc;
  const char* plaintext;
  const char* recvPrivKey;
  const char* authSecret;
  const char* payload;
  size_t payloadLen;
  size_t maxPlaintextLen;
  size_t plaintextLen;
} webpush_aes128gcm_decrypt_ok_test_t;

static webpush_aes128gcm_decrypt_ok_test_t
  webpush_aes128gcm_decrypt_ok_tests[] = {
    {
      .desc = "rs = 24, pad = 0",
      .plaintext = "I am the walrus",
      .recvPrivKey = "\xc8\x99\xd1\x1d\x32\xe2\xb7\xe6\xfe\x74\x98\x78\x6f\x50"
                     "\xf2\x3b\x98\xac\xe5\x39\x7a\xd2\x61\xde\x39\xba\x64\x49"
                     "\xec\xc1\x2c\xad",
      .authSecret =
        "\x99\x6f\xad\x8b\x50\xaa\x2d\x02\xb8\x3f\x26\x41\x2b\x2e\x2a\xee",
      .payload = "\x49\x5c\xe6\xc8\xde\x93\xa4\x53\x9e\x86\x2e\x86\x34\x99\x3c"
                 "\xbb\x00\x00"
                 "\x00\x18\x41\x04\x3c\x33\x78\xa2\xc0\xab\x95\x4e\x14\x98\x71"
                 "\x8e\x85\xf0"
                 "\x8b\xb7\x23\xfb\x7d\x25\xe1\x35\xa6\x63\xfe\x38\x58\x84\xeb"
                 "\x81\x92\x33"
                 "\x6b\xf9\x0a\x54\xed\x72\x0f\x1c\x04\x5c\x0b\x40\x5e\x9b\xbc"
                 "\x3a\x21\x42"
                 "\xb1\x6c\x89\x08\x67\x34\xc3\x74\xeb\xaf\x70\x99\xe6\x42\x7e"
                 "\x2d\x32\xc8"
                 "\xad\xa5\x01\x87\x03\xc5\x4b\x10\xb4\x81\xe1\x02\x7d\x72\x09"
                 "\xd8\xc6\xb4"
                 "\x35\x53\xfa\x13\x3a\xfa\x59\x7f\x2d\xdc\x45\xa5\xba\x81\x40"
                 "\x94\x4e\x64"
                 "\x90\xbb\x8d\x6d\x99\xba\x1d\x02\xe6\x0d\x95\xf4\x8c\xe6\x44"
                 "\x47\x7c\x17"
                 "\x23\x1d\x95\xb9\x7a\x4f\x95\xdd",
      .payloadLen = 152,
      .maxPlaintextLen = 18,
      .plaintextLen = 15,
    },
    {
      .desc = "rs = 49, pad = 84; ciphertext length falls on record boundary",
      .plaintext = "Hello, world",
      .recvPrivKey = "\x67\x00\x4a\x4e\xa8\x20\xde\xed\x8e\x49\xdb\x5e\x94\x80"
                     "\xe6\x3d\x3e\xa3\xcc\xe1\xae\x8e\x1a\x60\x60\x97\x13\xd5"
                     "\x27\xd0\x01\xef",
      .authSecret =
        "\x95\xf1\x75\x70\xe5\x08\xef\x6a\x2b\x2a\xd1\xb4\xf5\xca\xde\x33",
      .payload =
        "\xfb\x28\x83\xce\xc1\xc4\xfc\xad\xd6\xd1\x37\x1f\x6e\xa4\x91\xe0\x00"
        "\x00\x00\x31\x41\x04\x2d\x44\x1e\xe7\xf9\xff\x6a\x03\x29\xa6\x49\x27"
        "\xd0\x52\x4f\xdb\xe7\xb2\x2c\x6f\xb6\x5e\x10\xab\x4f\xdc\x03\x8f\x94"
        "\x42\x0a\x0c\xa3\xfa\x28\xda\xd3\x6c\x84\xec\x91\xa1\x62\xea\xe0\x78"
        "\xfa\xad\x2c\x1c\xed\x78\xde\x81\x13\xe1\x96\x02\xb2\x0e\x89\x4f\x49"
        "\x76\xb9\x73\xe2\xfc\xf6\x82\xfa\x0c\x8c\xcd\x9a\xf3\xd5\xbf\xf1\xed"
        "\xe1\x6f\xad\x5a\x31\xce\x19\xd3\x8b\x5e\x1f\xe1\xf7\x8a\x4f\xad\x84"
        "\x2b\xbc\x10\x25\x4c\x2c\x6c\xdd\x96\xa2\xb5\x52\x84\xd9\x72\xc5\x3c"
        "\xad\x8c\x3b\xac\xb1\x0f\x5f\x57\xeb\x0d\x4a\x43\x33\xb6\x04\x10\x2b"
        "\xa1\x17\xca\xe2\x91\x08\xfb\xd9\xf6\x29\xa8\xba\x69\x60\xdd\x01\x94"
        "\x5b\x39\xed\x37\xba\x70\x6c\x43\x4a\x10\xfd\x2b\xd2\x09\x4f\xf9\x24"
        "\x9b\xcd\xad\x45\x13\x5f\x5f\xe4\x5f\xcd\x38\x07\x1f\x8b\x2d\x39\x41"
        "\xaf\xda\x43\x98\x10\xd7\x7a\xac\xaf\x7c\xe5\x0b\x54\x32\x5b\xf5\x8c"
        "\x95\x03\x33\x7d\x07\x37\x85\xa3\x23\xdf\xa3\x43",
      .payloadLen = 233,
      .maxPlaintextLen = 99,
      .plaintextLen = 12,
    },
    {
      .desc = "Example from draft-ietf-webpush-encryption-latest",
      .plaintext = "When I grow up, I want to be a watermelon",
      .recvPrivKey = "\xab\x57\x57\xa7\x0d\xd4\xa5\x3e\x55\x3a\x6b\xbf\x71\xff"
                     "\xef\xea\x28\x74\xec\x07\xa6\xb3\x79\xe3\xc4\x8f\x89\x5a"
                     "\x02\xdc\x33\xde",
      .authSecret =
        "\x05\x30\x59\x32\xa1\xc7\xea\xbe\x13\xb6\xce\xc9\xfd\xa4\x88\x82",
      .payload = "\x0c\x6b\xfa\xad\xad\x67\x95\x88\x03\x09\x2d\x45\x46\x76\xf3"
                 "\x97\x00\x00\x10\x00\x41\x04\xfe\x33\xf4\xab\x0d\xea\x71\x91"
                 "\x4d\xb5\x58\x23\xf7\x3b\x54\x94\x8f\x41\x30\x6d\x92\x07\x32"
                 "\xdb\xb9\xa5\x9a\x53\x28\x64\x82\x20\x0e\x59\x7a\x7b\x7b\xc2"
                 "\x60\xba\x1c\x22\x79\x98\x58\x09\x92\xe9\x39\x73\x00\x2f\x30"
                 "\x12\xa2\x8a\xe8\xf0\x6b\xbb\x78\xe5\xec\x0f\xf2\x97\xde\x5b"
                 "\x42\x9b\xba\x71\x53\xd3\xa4\xae\x0c\xaa\x09\x1f\xd4\x25\xf3"
                 "\xb4\xb5\x41\x4a\xdd\x8a\xb3\x7a\x19\xc1\xbb\xb0\x5c\xf5\xcb"
                 "\x5b\x2a\x2e\x05\x62\xd5\x58\x63\x56\x41\xec\x52\x81\x2c\x6c"
                 "\x8f\xf4\x2e\x95\xcc\xb8\x6b\xe7\xcd",
      .payloadLen = 144,
      .maxPlaintextLen = 42,
      .plaintextLen = 41,
    },
    {
      .desc = "rs = 18, pad = 0",
      .plaintext = "1",
      .recvPrivKey = "\x27\x43\x3f\xab\x89\x70\xb3\xcb\x52\x84\xb6\x11\x83\xef"
                     "\xb4\x62\x86\x56\x2c\xd2\xa7\x33\x0d\x8c\xae\x96\x09\x11"
                     "\xa5\x57\x1d\x0c",
      .authSecret =
        "\xd6\x5a\x04\xdf\x95\xf2\xdb\x5e\x60\x48\x39\xf7\x17\xdc\xde\x79",
      .payload = "\x7c\xae\xbd\xbc\x20\x93\x8e\xe3\x40\xa9\x46\xf1\xbd\x4f\x68"
                 "\xf1\x00\x00\x00\x12\x41\x04\x37\xcf\xdb\x52\x23\xd9\xf9\x5e"
                 "\xaa\x02\xf6\xed\x94\x0f\xf2\x2e\xaf\x05\xb3\x62\x2e\x94\x9d"
                 "\xc3\xce\x9f\x33\x5e\x6e\xf9\xb2\x6a\xea\xac\xca\x0f\x74\x08"
                 "\x0a\x8b\x36\x45\x92\xf2\xcc\xc6\xd5\xed\xdd\x43\x00\x4b\x70"
                 "\xb9\x18\x87\xd1\x44\xd9\xfa\x93\xf1\x6c\x3b\xc7\xea\x68\xf4"
                 "\xfd\x54\x7a\x94\xec\xa8\x4b\x16\xe1\x38\xa6\x08\x01\x77",
      .payloadLen = 104,
      .maxPlaintextLen = 2,
      .plaintextLen = 1,
    },
};

typedef struct webpush_aes128gcm_err_decrypt_test_s {
  const char* desc;
  const char* recvPrivKey;
  const char* authSecret;
  const char* payload;
  size_t payloadLen;
  size_t maxPlaintextLen;
  int err;
} webpush_aes128gcm_err_decrypt_test_t;

static webpush_aes128gcm_err_decrypt_test_t
  webpush_aes128gcm_err_decrypt_tests[] = {
    {
      // Header block shorter than 21 bytes.
      .desc = "Missing header block",
      .recvPrivKey = "\x1b\xe8\x3f\x38\x33\x2e\xf0\x96\x81\xfa\xf3\xf3\x07\xb1"
                     "\xff\x2e\x10\xca\xb7\x8c\xc7\xcd\xab\x68\x3a\xc0\xee\x92"
                     "\xac\x3f\x6e\xe1",
      .authSecret =
        "\x34\x71\xbb\x98\x48\x1e\x02\x53\x3b\xf3\x95\x42\xbc\xf3\xdb\xa4",
      .payload = "\x45\xb7\x4d\x2b\x69\xbe\x9b\x07\x4d\xe3\xb3\x5a\xa8\x7e\x7c"
                 "\x15\x61\x1d",
      .payloadLen = 18,
      .maxPlaintextLen = 0,
      .err = ECE_ERROR_SHORT_HEADER,
    },
    {
      // Sender key shorter than 65 bytes.
      .desc = "Truncated sender key",
      .recvPrivKey = "\xce\x88\xe8\xe0\xb3\x05\x7a\x47\x52\xeb\x4c\x8f\xa9\x31"
                     "\xeb\x62\x1c\x30\x2d\xa5\xad\x03\xb8\x1a\xf4\x59\xcf\x67"
                     "\x35\x56\x0c\xae",
      .authSecret =
        "\x5c\x31\xe0\xd9\x6d\x9a\x13\x98\x99\xac\x09\x69\xd3\x59\xf7\x40",
      .payload = "\xde\x5b\x69\x6b\x87\xf1\xa1\x5c\xb6\xad\xeb\xdd\x79\xd6\xf9"
                 "\x9e\x00\x00\x00\x12\x01\x00\xb6\xbc\x18\x26\xc3\x7c\x9f\x73"
                 "\xdd\x6b\x48\x59\xc2\xb5\x05\x18\x19\x52",
      .payloadLen = 40,
      .maxPlaintextLen = 2,
      .err = ECE_ERROR_COMPUTE_SECRET,
    },
    {
      // The payload is encrypted with only the first 12 bytes of the auth
      // secret.
      .desc = "Truncated auth secret",
      .recvPrivKey = "\x60\xc7\x63\x6a\x51\x7d\xe7\x03\x9a\x0a\xc2\xd0\xe3\x06"
                     "\x44\x00\x79\x4c\x78\xe7\xe0\x49\x39\x81\x29\xa2\x27\xce"
                     "\xe0\xf9\xa8\x01",
      .authSecret =
        "\x35\x5a\x38\xcd\x6d\x9b\xef\x15\x99\x0e\x2d\x33\x08\xdb\xd6\x00",
      .payload = "\x81\x15\xf4\x98\x8b\x8c\x39\x2a\x7b\xac\xb4\x3c\x8f\x1a\xc5"
                 "\x65\x00\x00\x00\x12\x41\x04\x19\x94\x48\x3c\x54\x1e\x9b\xc3"
                 "\x9a\x6a\xf0\x3f\xf7\x13\xaa\x77\x45\xc2\x84\xe1\x38\xa4\x2a"
                 "\x24\x35\xb7\x97\xb2\x0c\x4b\x69\x8c\xf5\x11\x8b\x4f\x85\x55"
                 "\x31\x7c\x19\x0e\xab\xeb\xfa\xb7\x49\xc1\x64\xd3\xf6\xbd\xeb"
                 "\xe0\xd4\x41\x71\x91\x31\xa3\x57\xd8\x89\x0a\x13\xc4\xdb\xd4"
                 "\xb1\x6f\xf3\xdd\x5a\x83\xf7\xc9\x1a\xd6\xe0\x40\xac\x42\x73"
                 "\x0a\x7f\x0b\x3c\xd3\x24\x5e\x9f\x8d\x6f\xf3\x1c\x75\x1d\x41"
                 "\x0c\xfd",
      .payloadLen = 122,
      .maxPlaintextLen = 4,
      .err = ECE_ERROR_DECRYPT,
    },
    {
      .desc = "Early final record",
      .recvPrivKey = "\x5d\xda\x1d\x91\x8b\xc4\x07\xba\x3c\xda\x12\xcb\x80\x14"
                     "\xd4\x9a\xa7\xe0\x26\x90\x02\x82\x03\x04\x46\x6b\xc8\x00"
                     "\x34\xca\x92\x40",
      .authSecret =
        "\x40\xc2\x41\xfd\xe4\x26\x9e\xe1\xe6\xd7\x25\x59\x2d\x98\x27\x18",
      .payload = "\xdb\xe2\x15\x50\x7d\x1a\xd3\xd2\xea\xea\xbe\xae\x6e\x87\x4d"
                 "\x8f\x00\x00\x00\x12\x41\x04\x7b\xc4\x34\x3f\x34\xa8\x34\x8c"
                 "\xdc\x4e\x46\x2f\xfc\x7c\x40\xaa\x6a\x8c\x61\xa7\x39\xc4\xc4"
                 "\x1d\x45\x12\x55\x05\xf7\x0e\x9f\xc5\xf9\xef\xa8\x68\x52\xdd"
                 "\x48\x8d\xcf\x8e\x8e\xa2\xca\xfb\x75\xe0\x7a\xbd\x5e\xe7\xc9"
                 "\xd5\xc0\x38\xba\xfe\xf0\x79\x57\x1b\x0b\xda\x29\x44\x11\xce"
                 "\x98\xc7\x6d\xd0\x31\xc0\xe5\x80\x57\x7a\x49\x80\xa3\x75\xe4"
                 "\x5e\xd3\x04\x29\xbe\x0e\x2e\xe9\xda\x7e\x6d\xf8\x69\x6d\x01"
                 "\xb8\xec",
      .payloadLen = 122,
      .maxPlaintextLen = 4,
      .err = ECE_ERROR_DECRYPT_PADDING,
    },
};

typedef struct aes128gcm_ok_decrypt_test_s {
  const char* desc;
  const char* plaintext;
  const char* ikm;
  const char* payload;
  size_t payloadLen;
  size_t maxPlaintextLen;
  size_t plaintextLen;
} aes128gcm_ok_decrypt_test_t;

static aes128gcm_ok_decrypt_test_t aes128gcm_ok_decrypt_tests[] = {
  {
    .desc = "rs = 18, pad = 8",
    .plaintext = "When I grow up, I want to be a watermelon",
    .ikm = "\x28\xc0\x66\x11\x4a\x2d\xa5\x21\xca\x89\xf4\x21\x9d\xa8\xac\xc0",
    .payload =
      "\x1f\xc2\xec\x59\x4d\xbd\xa8\xc8\xab\x26\x25\x47\x04\x65\xb8\xcd\x00\x00"
      "\x00\x12\x00\x92\x56\xfe\x1c\x43\x4f\x71\x8e\x85\x16\x3a\x0f\x52\x69\xc1"
      "\xb8\x24\x55\x73\x60\x7d\x06\x06\xc3\x97\xfc\xfd\xc3\x27\xd5\xf9\x0c\x44"
      "\x8d\x6a\x11\xa0\xc4\xb8\xd0\x51\xc8\x54\x94\xb0\x0f\xb5\xeb\xb9\xe6\x85"
      "\x38\x2f\x88\xee\x5a\xce\x19\x1b\xfa\x73\x1d\xa2\xc9\xb2\x3f\x0a\xe4\xfe"
      "\x4b\x9a\xd5\xf5\x4d\xf0\xec\xc8\x17\x9f\xc6\xdb\xed\x3a\x94\x33\xbe\x4f"
      "\x92\xa8\xdd\xf1\x0d\x5f\x29\x9f\x76\x73\xfb\x79\x33\x69\xc9\x6b\xf5\x20"
      "\x5b\x4e\xa5\x47\xef\xa3\xd4\x4b\x6c\xaa\x47\xac\x97\x9a\xa1\x69\x45\x2a"
      "\xf6\xf6\x84\x65\xda\xba\x9b\x8a\xb3\x9c\xed\x91\x15\xd4\x4f\xbb\x7c\xf6"
      "\xc6\xfa\x0f\x86\x71\xa2\xa1\x2c\xf6\x18\x18\x86\x94\xf1\x7c\x2f\x63\xb7"
      "\x46\xe0\x6e\x9a\x51\x20\x6a\x8c\x54\xc9\x91\x54\xb1\x84\xa9\xec\x8a\x29"
      "\x71\x4e\xfd\xb6\x8f\xde\xe4\xc4\x2f\x57\xb3\x2e\x48\x8d\x5c\x47\x51\x05"
      "\xd0\x57\xb6\x55\x15\xc4\xa0\xeb\x59\x5b\xd6\xe8\xa7\x11\x65\x18\xad\xfb"
      "\xc5\xdf\xbc\x51\x71\x01\xae\x72\x2b\x19\x14\xa1\x47\x3e\x35\xbb\x52\xa7"
      "\xc9\xad\x22\x09\xc6\xea\x8f\x2b\x60\x5f\x8d\xf9\x78\x65\x4e\xd5\x2c\x71"
      "\x17\x5c\x12\x4e\xb3\xa5\x6e\xfa\xfe\x64\x77\xd8\x05\x07\x4d\xd0\x29\x15"
      "\x65\x37\x4b\xb1\x02\x8c\x9b\xbd\x59\xd6\x4d\x56\x27\xe7\x28\x02\x5a\x30"
      "\x59\x10\x0f\x48\xe8\x88\x5f\x7f\xe4\x4e\x0d\xec\xbb\x7b\x98\x08\x3d\x85"
      "\xe1\xc8\x2b\x11\x8a\x8a\xf5\xb3\x8f\x33\xb3\xb6\x7b\xa6\xd5\x8e\x58\xc1"
      "\x3a\xff\xaf\x8c\x2b\x99\x9d\x4f\xc2\x09\xed\x73\x7a\x04\x74\x93\x48\x1b"
      "\xfa\xfd\x71\x9d\x49\x8d\xf0\xa9\x8c\x6f\x43\x48\xc7\x24\x3a\xa6\x78\x9a"
      "\xf3\x36\x85\x4b\x7e\x87\xf9\x5a\x05\x47\xcf\x73\x5e\xc3\x83\xa8\x27\x4d"
      "\xdc\xf5\xd9\x76\x43\x85\x37\x36\xb4\xc6\x06\x3f\x48\x95\xab\x38\x38\xc9"
      "\x99\x9c\xec\x7b\x73\x1c\xda\xcb\xd5\x0f\x8c\x06\xde\x9f\xe8\x0a\xad\xaf"
      "\x91\xd1\x1b\x9a\x35\xaa\xdf\x41\xa0\x5c\x6b\xae\xda\x0c\x6d\x00\xb4\xa8"
      "\xc0\xc3\x69\xf9\x8f\x4c\x6e\x6a\x97\x76\xab\x41\x7e\x28\x39\x1b\x47\x5c"
      "\xe7\xfc\x01\x65\xdb\xe4\x9e\xf9\x89\x1f\x9c\xef\x82\xe2\x86\x7e\xd6\xd6"
      "\x7c\x4a\x5a\x71\xda\xa1\xf7\x5d\x26\x5f\x85\x92\xa8\x1d\xb4\x8c\xbc\x92"
      "\xc3\x82\xd6\x3a\x96\xf5\x80\x0f\xa8\xec\xa9\xe2\x02\x7b\xaf\xb7\x4b\xc9"
      "\xe3\x3b\xdc\xd3\xb8\xf4\xd8\xe0\x5f\x36\xdd\xa5\x44\xf8\x97\x5e\xcb\xea"
      "\x47\x8d\xb8\x36\x61\xa1\xdb\xc5\xfc\xcb\x7f\xeb\x05\x57\xde\xd7\x3a\x37"
      "\x90\xc3\x52\x69\xfa\x59\xe4\x75\x0e\x55\xc7\x29\xa0\x08\xc9\x8c\xe9\xee"
      "\x88\x82\xe0\xc2\xae\xaf\x1e\xbe\x40\x3b\xe9\x6d\xaa\x25\xb4\x2a\xc0\x1b"
      "\x6a\xd4\x35\x5b\xc3\x60\xcd\xd1\x31\x10\xe3\xff\xc7\x6a\xb4\x51\xf5\x9e"
      "\x04\xa8\xab\x3f\x1a\x4a\x69\xdf\x21\x91\xab\x4b\x60\xfd\x31\x76\x13\x2b"
      "\x8f\x99\x1d\x3a\xb2\x96\xa0\x36\x93\x36\xb5\x35\xaa\xcc\x15\x97\x7d\x50"
      "\x5b\xe2\xc5\xd4\xb6\xb7\xbc\x55\xd8\x3c\xd1\x7c\x1e\x80\x35\x7f\x4a\x21"
      "\x8d\x72\x93\x3f\xa6\x09\x74\x73\x29\x6d\x7d\xdc\x30\xd7\xa1\x7b\x73\x23"
      "\xac\x17\x3e\xc3\x47\x72\x45\x1a\xa6\x70\x95\xca\xcf\xbc\x87\x6c\x05\x56"
      "\xa7\xae\x2f\x2c\x64\x55\x50\xd8\xab\x05\xe6\xa7\x87\xa5\x5a\x1c\x0c\xe1"
      "\x59\x7b\x95\x8e\xe7\xea\xff\x10\x29\x93\x02\xd9\x9c\x35\xca\xc9\x83\xb9"
      "\x6c\x0f\xec\xaf\x61\x59\x3a\x17\x66\xd3\xbc\xc7\xc2\xd5\x00\x4a\x4c\x43"
      "\x91\xcb\x41\xb1\x36\x79\x35\xe3\x9d\x73\xf3\xa2\xe0\x84\x59\xd2\x83\x2c"
      "\x18\x34\xf2\x60\x6a\x87\x40\x10\xd7\xcf\x17\x7e\x7b\xcf\x61\xad\x41\x3d"
      "\x0f\xfc\xe3\x3d\x6b\xef\x39\xb9\x61\x39\xda\x24\xaf\xc9\xac\xe7\x94\x28"
      "\x8d\x79\x75\xac\x74\xc9\x86\x66\x1d\x40\x34\x42\x25\xf7\x99\xf5\x96\x35"
      "\xa9\x1f\x98\x7b\x54\x42\x3a\x5e\x10\x60\x9b\x6d\x8e\xb4\xda\xe5\xc2\xc8"
      "\x2e\x53\xae\xc3\xa3\xdd\xe5\xaf\xbf\x06\x2c\x42\xe2\x95\x91\xd9\x3e\x49"
      "\xe4\x54\x80\x90\xb5\x22\x7e\x13\xda\x62\x70\x14\x7e\x5d\x9d\xee\x3f\x2e"
      "\x8d\x2f\x7d\xe0\xaf\x1d\xd7\x61\x27\x9d\xd3\xf2\x8a\xce\x13\x74\x73\x15"
      "\x11\xf2\x1a",
    .payloadLen = 903,
    .maxPlaintextLen = 98,
    .plaintextLen = 41,
  },
};

typedef struct aes128gcm_err_decrypt_test_s {
  const char* desc;
  const char* ikm;
  const char* payload;
  size_t payloadLen;
  size_t maxPlaintextLen;
  int err;
} aes128gcm_err_decrypt_test_t;

static aes128gcm_err_decrypt_test_t aes128gcm_err_decrypt_tests[] = {
  {
    .desc = "Truncated ciphertext, rs = 18",
    .ikm = "\x28\xc0\x66\x11\x4a\x2d\xa5\x21\xca\x89\xf4\x21\x9d\xa8\xac\xc0",
    .payload = "\x1f\xc2\xec\x59\x4d\xbd\xa8\xc8\xab\x26\x25\x47\x04\x65\xb8"
               "\xcd\x00\x00\x00\x12\x00\x92\x56\xfe\x1c\x43\x4f\x71\x8e\x85"
               "\x16\x3a\x0f\x52\x69\xc1\xb8\x24\x55",
    .payloadLen = 40,
    .maxPlaintextLen = 0,
    .err = ECE_ERROR_DECRYPT,
  },
  {
    .desc = "rs <= block overhead",
    .ikm = "\x2f\xb1\x75\xc2\x71\xb9\x2f\x6b\x55\xe4\xf2\xa2\x52\xd1\x45\x43",
    .payload = "\x76\xf9\x1d\x48\x4e\x84\x91\xda\x55\xc5\xf7\xbf\xe6\xd3\x3e"
               "\x89\x00\x00\x00\x02\x00",
    .payloadLen = 21,
    .maxPlaintextLen = 0,
    .err = ECE_ERROR_INVALID_RS,
  },
  {
    .desc = "Zero plaintext",
    .ikm = "\x64\xc7\x0e\x64\xa7\x25\x55\x14\x51\xf2\x08\xdf\xba\xa0\xb9\x72",
    .payload = "\xaa\xd2\x05\x7d\x33\x53\xb7\xff\x37\xbd\xe4\x2a\xe1\xd5\x0f"
               "\xda\x00\x00\x00\x20\x00\xbb\xc7\xb9\x65\x76\x0b\xf0\x66\x2b"
               "\x93\xf4\xe5\xd6\x94\xb7\x65\xf0\xcd\x15\x9b\x28\x01\xa5",
    .payloadLen = 44,
    .maxPlaintextLen = 7,
    .err = ECE_ERROR_ZERO_PLAINTEXT,
  },
  {
    .desc = "Bad early padding delimiter",
    .ikm = "\x64\xc7\x0e\x64\xa7\x25\x55\x14\x51\xf2\x08\xdf\xba\xa0\xb9\x72",
    .payload = "\xaa\xd2\x05\x7d\x33\x53\xb7\xff\x37\xbd\xe4\x2a\xe1\xd5\x0f"
               "\xda\x00\x00\x00\x20\x00\xb9\xc7\xb9\x65\x76\x0b\xf0\x9e\x42"
               "\xb1\x08\x43\x38\x75\xa3\x06\xc9\x78\x06\x0a\xfc\x7c\x7d\xe9"
               "\x52\x85\x91\x8b\x58\x02\x60\xf3\x45\x38\x7a\x28\xe5\x25\x66"
               "\x2f\x48\xc1\xc3\x32\x04\xb1\x95\xb5\x4e\x9e\x70\xd4\x0e\x3c"
               "\xf3\xef\x0c\x67\x1b\xe0\x14\x49\x7e\xdc",
    .payloadLen = 85,
    .maxPlaintextLen = 32,
    .err = ECE_ERROR_DECRYPT_PADDING,
  },
  {
    .desc = "Bad final padding delimiter",
    .ikm = "\x64\xc7\x0e\x64\xa7\x25\x55\x14\x51\xf2\x08\xdf\xba\xa0\xb9\x72",
    .payload = "\xaa\xd2\x05\x7d\x33\x53\xb7\xff\x37\xbd\xe4\x2a\xe1\xd5\x0f"
               "\xda\x00\x00\x00\x20\x00\xba\xc7\xb9\x65\x76\x0b\xf0\x9e\x42"
               "\xb1\x08\x4a\x69\xe4\x50\x1b\x8d\x49\xdb\xc6\x79\x23\x4d\x47"
               "\xc2\x57\x16",
    .payloadLen = 48,
    .maxPlaintextLen = 11,
    .err = ECE_ERROR_DECRYPT_PADDING,
  },
  {
    .desc = "Invalid auth tag",
    .ikm = "\x64\xc7\x0e\x64\xa7\x25\x55\x14\x51\xf2\x08\xdf\xba\xa0\xb9\x72",
    .payload = "\xaa\xd2\x05\x7d\x33\x53\xb7\xff\x37\xbd\xe4\x2a\xe1\xd5\x0f"
               "\xda\x00\x00\x00\x20\x00\xbb\xc6\xb1\x1d\x46\x3a\x7e\x0f\x07"
               "\x2b\xbe\xaa\x44\xe0\xd6\x2e\x4b\xe5\xf9\x5d\x25\xe3\x86\x71"
               "\xe0\x7d",
    .payloadLen = 47,
    .maxPlaintextLen = 10,
    .err = ECE_ERROR_DECRYPT,
  },
  {
    // 2 records; last record is "\x00" without a delimiter.
    .desc = "rs = 21, truncated padding for last record",
    .ikm = "\x1a\x5c\x05\x64\x16\xdf\x83\x73\x87\x51\x01\xd1\x11\x98\x47\x83",
    .payload = "\x53\x06\xdc\x45\xdd\x8e\x51\x00\x16\x53\x3c\x1e\xba\xe5\x50"
               "\x53\x00\x00\x00\x15\x00\xa7\x0d\x92\x4e\xe6\x08\xd0\xc1\xc1"
               "\x00\x88\x5a\xe8\x78\x1d\xd1\x47\x67\x02\x12\x63\xf7\x9d\x22"
               "\xa9\x44\x8d\xb2\x33\x6e\xe0\xe5\x72\xe2\x3c\x38\x49\x70",
    .payloadLen = 59,
    .maxPlaintextLen = 6,
    .err = ECE_ERROR_ZERO_PLAINTEXT,
  },
  {
    // 2 records; last record is just the auth tag.
    .desc = "rs = 21, auth tag for last record",
    .ikm = "\xc1\xc9\xc0\x91\x9d\x81\x0a\xe7\xd9\xe8\x0c\x45\xbc\x21\xa9\xfa",
    .payload = "\xc1\xaf\x29\x07\x6f\x69\x25\x60\xde\x6d\x1f\xde\x02\x11\x69"
               "\x79\x00\x00\x00\x15\x00\x46\x9f\xde\x73\xa7\x8a\x2a\x66\x1d"
               "\xb0\xf1\xae\x55\xec\xec\x86\x6a\xaa\xe5\xf3\x04\xa3\x3e\xc3"
               "\xb0\xbb\x16\xe9\x0a\xab\xc4\xba\xe0\xed\xbb\x73\x46",
    .payloadLen = 58,
    .maxPlaintextLen = 5,
    .err = ECE_ERROR_SHORT_BLOCK,
  },
};

void
test_webpush_aes128gcm_decrypt_ok(void) {
  size_t tests = sizeof(webpush_aes128gcm_decrypt_ok_tests) /
                 sizeof(webpush_aes128gcm_decrypt_ok_test_t);
  for (size_t i = 0; i < tests; i++) {
    webpush_aes128gcm_decrypt_ok_test_t t =
      webpush_aes128gcm_decrypt_ok_tests[i];

    const void* recvPrivKey = t.recvPrivKey;
    const void* authSecret = t.authSecret;
    const void* payload = t.payload;

    size_t plaintextLen =
      ece_aes128gcm_plaintext_max_length(payload, t.payloadLen);
    ece_assert(plaintextLen == t.maxPlaintextLen,
               "Got plaintext max length %zu for `%s`; want %zu", plaintextLen,
               t.desc, t.maxPlaintextLen);

    uint8_t* plaintext = calloc(plaintextLen, sizeof(uint8_t));

    int err = ece_webpush_aes128gcm_decrypt(
      recvPrivKey, ECE_WEBPUSH_PRIVATE_KEY_LENGTH, authSecret,
      ECE_WEBPUSH_AUTH_SECRET_LENGTH, payload, t.payloadLen, plaintext,
      &plaintextLen);
    ece_assert(!err, "Got %d decrypting payload for `%s`", err, t.desc);

    ece_assert(plaintextLen == t.plaintextLen,
               "Got plaintext length %zu for `%s`; want %zu", plaintextLen,
               t.desc, t.plaintextLen);
    ece_assert(!memcmp(plaintext, t.plaintext, plaintextLen),
               "Wrong plaintext for `%s`", t.desc);

    free(plaintext);
  }
}

void
test_webpush_aes128gcm_decrypt_err(void) {
  size_t tests = sizeof(webpush_aes128gcm_err_decrypt_tests) /
                 sizeof(webpush_aes128gcm_err_decrypt_test_t);
  for (size_t i = 0; i < tests; i++) {
    webpush_aes128gcm_err_decrypt_test_t t =
      webpush_aes128gcm_err_decrypt_tests[i];

    const void* recvPrivKey = t.recvPrivKey;
    const void* authSecret = t.authSecret;
    const void* payload = t.payload;

    size_t plaintextLen =
      ece_aes128gcm_plaintext_max_length(payload, t.payloadLen);
    ece_assert(plaintextLen == t.maxPlaintextLen,
               "Got plaintext max length %zu for `%s`; want %zu", plaintextLen,
               t.desc, t.maxPlaintextLen);

    uint8_t* plaintext = calloc(plaintextLen, sizeof(uint8_t));

    int err = ece_webpush_aes128gcm_decrypt(
      recvPrivKey, ECE_WEBPUSH_PRIVATE_KEY_LENGTH, authSecret,
      ECE_WEBPUSH_AUTH_SECRET_LENGTH, payload, t.payloadLen, plaintext,
      &plaintextLen);
    ece_assert(err == t.err, "Got %d decrypting payload for `%s`; want %d", err,
               t.desc, t.err);

    free(plaintext);
  }
}

void
test_aes128gcm_decrypt_ok(void) {
  size_t tests =
    sizeof(aes128gcm_ok_decrypt_tests) / sizeof(aes128gcm_ok_decrypt_test_t);
  for (size_t i = 0; i < tests; i++) {
    aes128gcm_ok_decrypt_test_t t = aes128gcm_ok_decrypt_tests[i];

    size_t plaintextLen = ece_aes128gcm_plaintext_max_length(
      (const uint8_t*) t.payload, t.payloadLen);
    ece_assert(plaintextLen == t.maxPlaintextLen,
               "Got plaintext max length %zu for `%s`; want %zu", plaintextLen,
               t.desc, t.maxPlaintextLen);

    uint8_t* plaintext = calloc(plaintextLen, sizeof(uint8_t));

    int err = ece_aes128gcm_decrypt((const uint8_t*) t.ikm, 16,
                                    (const uint8_t*) t.payload, t.payloadLen,
                                    plaintext, &plaintextLen);
    ece_assert(!err, "Got %d decrypting payload for `%s`", err, t.desc);

    ece_assert(plaintextLen == t.plaintextLen,
               "Got plaintext length %zu for `%s`; want %zu", plaintextLen,
               t.desc, t.plaintextLen);
    ece_assert(!memcmp(plaintext, t.plaintext, plaintextLen),
               "Wrong plaintext for `%s`", t.desc);

    free(plaintext);
  }
}

void
test_aes128gcm_decrypt_err(void) {
  size_t tests =
    sizeof(aes128gcm_err_decrypt_tests) / sizeof(aes128gcm_err_decrypt_test_t);
  for (size_t i = 0; i < tests; i++) {
    aes128gcm_err_decrypt_test_t t = aes128gcm_err_decrypt_tests[i];

    const void* ikm = t.ikm;
    const void* payload = t.payload;

    size_t plaintextLen =
      ece_aes128gcm_plaintext_max_length(payload, t.payloadLen);
    ece_assert(plaintextLen == t.maxPlaintextLen,
               "Got plaintext max length %zu for `%s`; want %zu", plaintextLen,
               t.desc, t.maxPlaintextLen);

    uint8_t* plaintext = calloc(plaintextLen, sizeof(uint8_t));

    int err = ece_aes128gcm_decrypt(ikm, 16, payload, t.payloadLen, plaintext,
                                    &plaintextLen);
    ece_assert(err == t.err, "Got %d decrypting payload for `%s`; want %d", err,
               t.desc, t.err);

    free(plaintext);
  }
}
