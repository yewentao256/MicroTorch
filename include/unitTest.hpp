/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#pragma once

#include "storage.hpp"

namespace microtorch {

const char* test_device();
const char* test_allocator(const Device& device);
const char* test_func_traits();
const char* test_func_ref();

int unit_test();
}  // namespace microtorch
