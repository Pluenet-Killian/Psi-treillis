// SPDX-License-Identifier: Apache-2.0
// view_window.cpp - Win32 window and display surface of the psi_view comparator.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "view_window.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

HWND g_wnd = nullptr;
HDC g_mem = nullptr;
HBITMAP g_dib = nullptr;
uint32_t* g_pix = nullptr;
int g_w = 0, g_h = 0;
HFONT g_font[2] = {};
ViewInput g_in;
bool g_drag = false;
int g_lx = 0, g_ly = 0;

LRESULT CALLBACK proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CLOSE:
            g_in.quit = true;
            DestroyWindow(h);
            return 0;
        case WM_DESTROY:
            g_in.quit = true;
            g_wnd = nullptr;
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (g_in.key_count < 32) g_in.keys[g_in.key_count++] = (int)w;
            return 0;
        case WM_LBUTTONDOWN:
            g_drag = true;
            g_lx = (short)LOWORD(l);
            g_ly = (short)HIWORD(l);
            SetCapture(h);
            return 0;
        case WM_LBUTTONUP:
            g_drag = false;
            ReleaseCapture();
            return 0;
        case WM_MOUSEMOVE: {
            const int x = (short)LOWORD(l), y = (short)HIWORD(l);
            g_in.mouse_x = x;
            g_in.mouse_y = y;
            if (g_drag) {
                g_in.drag_dx += (float)(x - g_lx);
                g_in.drag_dy += (float)(y - g_ly);
                g_lx = x;
                g_ly = y;
            }
            return 0;
        }
        case WM_MOUSEWHEEL:
            g_in.wheel += (float)GET_WHEEL_DELTA_WPARAM(w) / (float)WHEEL_DELTA;
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC dc = BeginPaint(h, &ps);
            BitBlt(dc, 0, 0, g_w, g_h, g_mem, 0, 0, SRCCOPY);
            EndPaint(h, &ps);
            return 0;
        }
    }
    return DefWindowProcW(h, m, w, l);
}

std::wstring widen(const char* utf8) {
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    std::wstring s(len > 0 ? len - 1 : 0, L'\0');
    if (len > 1) MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &s[0], len);
    return s;
}

uint32_t g_crc[256];
void crc_init() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        g_crc[i] = c;
    }
}
uint32_t crc_of(const uint8_t* p, size_t n, uint32_t c) {
    for (size_t i = 0; i < n; ++i) c = g_crc[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}
void chunk(FILE* f, const char type[4], const std::vector<uint8_t>& data) {
    std::vector<uint8_t> head;
    put32(head, (uint32_t)data.size());
    fwrite(head.data(), 1, 4, f);
    fwrite(type, 1, 4, f);
    if (!data.empty()) fwrite(data.data(), 1, data.size(), f);
    uint32_t c = crc_of((const uint8_t*)type, 4, 0xFFFFFFFFu);
    if (!data.empty()) c = crc_of(data.data(), data.size(), c);
    std::vector<uint8_t> tail;
    put32(tail, c ^ 0xFFFFFFFFu);
    fwrite(tail.data(), 1, 4, f);
}

}  // namespace

bool view_open(int width, int height, const char* title_utf8, bool visible) {
    SetProcessDPIAware();
    g_w = width;
    g_h = height;
    HDC screen = GetDC(nullptr);
    g_mem = CreateCompatibleDC(screen);
    ReleaseDC(nullptr, screen);
    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = width;
    bi.bmiHeader.biHeight = -height;  // row 0 at the top
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    g_dib = CreateDIBSection(g_mem, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!g_dib || !bits) return false;
    g_pix = (uint32_t*)bits;
    SelectObject(g_mem, g_dib);
    SetBkMode(g_mem, TRANSPARENT);
    g_font[0] = CreateFontW(-17, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                            FIXED_PITCH | FF_MODERN, L"Consolas");
    g_font[1] = CreateFontW(-22, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                            VARIABLE_PITCH | FF_SWISS, L"Segoe UI");
    crc_init();
    if (!visible) return true;

    WNDCLASSW wc = {};
    wc.lpfnWndProc = proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // standard arrow
    wc.lpszClassName = L"PsiTreillisView";
    RegisterClassW(&wc);
    const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT r = {0, 0, width, height}, wa = {0, 0, 0, 0};
    AdjustWindowRect(&r, style, FALSE);
    // At the top of the work area, centred: the whole window stays visible.
    int px = CW_USEDEFAULT, py = CW_USEDEFAULT;
    if (SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0)) {
        px = wa.left + std::max(0, (int)((wa.right - wa.left) - (r.right - r.left)) / 2);
        py = wa.top;
    }
    g_wnd = CreateWindowW(wc.lpszClassName, widen(title_utf8).c_str(), style, px, py, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance,
                          nullptr);
    if (!g_wnd) return false;
    ShowWindow(g_wnd, SW_SHOW);
    return true;
}

uint32_t* view_pixels() {
    GdiFlush();
    return g_pix;
}

void view_poll(ViewInput& in) {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    in = g_in;
    g_in.key_count = 0;
    g_in.drag_dx = g_in.drag_dy = 0.0f;
    g_in.wheel = 0.0f;
}

void view_panel(int x, int y, int w, int h, float keep) {
    GdiFlush();
    const int k = (int)(std::min(1.0f, std::max(0.0f, keep)) * 256.0f);
    const int x0 = std::max(0, x), y0 = std::max(0, y), x1 = std::min(g_w, x + w), y1 = std::min(g_h, y + h);
    for (int j = y0; j < y1; ++j) {
        uint32_t* row = g_pix + (size_t)j * g_w;
        for (int i = x0; i < x1; ++i) {
            const uint32_t p = row[i];
            row[i] = ((((p >> 16) & 0xFF) * k >> 8) << 16) | ((((p >> 8) & 0xFF) * k >> 8) << 8) | ((p & 0xFF) * k >> 8);
        }
    }
}

void view_text(int x, int y, uint32_t rgb, const char* utf8, int size) {
    const std::wstring s = widen(utf8);
    SelectObject(g_mem, g_font[size ? 1 : 0]);
    SetTextColor(g_mem, RGB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF));
    TextOutW(g_mem, x, y, s.c_str(), (int)s.size());
}

void view_line(int x0, int y0, int x1, int y1, uint32_t rgb) {
    SelectObject(g_mem, GetStockObject(DC_PEN));
    SetDCPenColor(g_mem, RGB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF));
    MoveToEx(g_mem, x0, y0, nullptr);
    LineTo(g_mem, x1, y1);
}

void view_rect(int x, int y, int w, int h, uint32_t rgb) {
    GdiFlush();
    const int x0 = std::max(0, x), y0 = std::max(0, y), x1 = std::min(g_w, x + w), y1 = std::min(g_h, y + h);
    for (int j = y0; j < y1; ++j)
        for (int i = x0; i < x1; ++i) g_pix[(size_t)j * g_w + i] = rgb;
}

namespace {
const DWORD kStyle = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
}

void view_fit(int& width, int& height) {
    SetProcessDPIAware();
    RECT wa, r = {0, 0, 0, 0};
    if (!SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0)) return;
    AdjustWindowRect(&r, kStyle, FALSE);
    width = std::min(width, (int)(wa.right - wa.left) - (int)(r.right - r.left));
    height = std::min(height, (int)(wa.bottom - wa.top) - (int)(r.bottom - r.top));
}

void view_present() {
    GdiFlush();
    if (!g_wnd) return;
    HDC dc = GetDC(g_wnd);
    BitBlt(dc, 0, 0, g_w, g_h, g_mem, 0, 0, SRCCOPY);
    ReleaseDC(g_wnd, dc);
}

// 8-bit RGB PNG, zlib stream in "stored" blocks (no compression): no codec required.
bool view_save_png(const char* path) {
    GdiFlush();
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    const size_t row = 1 + (size_t)3 * g_w;
    std::vector<uint8_t> raw(row * g_h);
    for (int y = 0; y < g_h; ++y) {
        uint8_t* d = raw.data() + row * y;
        *d++ = 0;
        const uint32_t* s = g_pix + (size_t)y * g_w;
        for (int x = 0; x < g_w; ++x) {
            *d++ = (uint8_t)(s[x] >> 16);
            *d++ = (uint8_t)(s[x] >> 8);
            *d++ = (uint8_t)s[x];
        }
    }
    std::vector<uint8_t> z;
    z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    z.push_back(0x78);
    z.push_back(0x01);
    for (size_t pos = 0; pos < raw.size();) {
        const size_t len = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + len == raw.size() ? 1 : 0);
        z.push_back((uint8_t)(len & 0xFF));
        z.push_back((uint8_t)(len >> 8));
        z.push_back((uint8_t)(~len & 0xFF));
        z.push_back((uint8_t)((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) {
        a = (a + v) % 65521u;
        b = (b + a) % 65521u;
    }
    put32(z, (b << 16) | a);
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr;
    put32(ihdr, (uint32_t)g_w);
    put32(ihdr, (uint32_t)g_h);
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(f, "IHDR", ihdr);
    chunk(f, "IDAT", z);
    chunk(f, "IEND", {});
    fclose(f);
    return true;
}

void view_close() {
    if (g_wnd) DestroyWindow(g_wnd);
    g_wnd = nullptr;
    if (g_dib) DeleteObject(g_dib);
    if (g_mem) DeleteDC(g_mem);
    for (HFONT& h : g_font)
        if (h) DeleteObject(h);
    g_dib = nullptr;
    g_mem = nullptr;
    g_pix = nullptr;
}
