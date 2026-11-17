// SPDX-License-Identifier: Apache-2.0
// view_window.h - display surface and window of the psi_view comparator.
//
// No external dependency: Win32 window, text and lines by GDI, PNG capture without compression. The rendering itself
// is done by CUDA kernels in a buffer that the caller copies into view_pixels().
#pragma once

#include <cstdint>

struct ViewInput {
    bool quit = false;
    int key_count = 0;
    int keys[32] = {};        // virtual-key codes of the keys pressed since the last call
    int mouse_x = 0, mouse_y = 0;
    float drag_dx = 0.0f, drag_dy = 0.0f;  // movement of the mouse, left button held down
    float wheel = 0.0f;                    // wheel notches
};

// Creates the surface (width x height, 32 bits) and, if visible, the window that displays it.
bool view_open(int width, int height, const char* title_utf8, bool visible);
// Pixels 0x00RRGGBB, row 0 at the top.
uint32_t* view_pixels();
void view_poll(ViewInput& in);
// Darkens a rectangle (factor between 0 and 1), to put readable text on the image.
void view_panel(int x, int y, int w, int h, float keep);
// size: 0 = fixed-width body text, 1 = title.
void view_text(int x, int y, uint32_t rgb, const char* utf8, int size = 0);
void view_line(int x0, int y0, int x1, int y1, uint32_t rgb);
void view_rect(int x, int y, int w, int h, uint32_t rgb);
// Reduces the requested size to what the screen can display (work area minus the frame of the window).
void view_fit(int& width, int& height);
void view_present();
bool view_save_png(const char* path);
void view_close();
