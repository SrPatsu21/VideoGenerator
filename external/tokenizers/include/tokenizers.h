#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* TokenizerHandle;

TokenizerHandle tokenizer_load(const char* path);
char* tokenizer_encode(TokenizerHandle handle, const char* text);
void tokenizer_free_string(char* s);
void tokenizer_destroy(TokenizerHandle handle);

#ifdef __cplusplus
}
#endif
