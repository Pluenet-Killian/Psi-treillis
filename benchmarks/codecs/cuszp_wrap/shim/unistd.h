// SPDX-License-Identifier: Apache-2.0
// Substitute for <unistd.h> with MSVC (the cuSZp tool uses only access() from it).
#pragma once
#include <io.h>
#ifndef F_OK
#define F_OK 0
#endif
#define access _access
