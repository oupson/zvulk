const std = @import("std");

const log = std.log.scoped(.windows);

const W = std.unicode.utf8ToUtf16LeStringLiteral;

const windows = @cImport({
    @cDefine("MIDL_INTERFACE", "struct");
    @cInclude("Windows.h");
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_win32.h");
});

const Wsi = @import("../wsi.zig");
pub const KeyboardEvent = Wsi.KeyboardEvent;

const Self = @This();

const Allocator = std.mem.Allocator;

pub const Listener = Wsi.Listener;
pub const ListenerVTable = Wsi.ListenerVtable;

allocator: Allocator,
hInstance: windows.HINSTANCE = null,
hWnd: windows.HWND = null,
listener: Wsi.Listener,
wRect: windows.RECT = .{},
have_focus: bool = false,

const required_extensions: [1]*align(1) const [:0]u8 = .{
    @ptrCast(windows.VK_KHR_WIN32_SURFACE_EXTENSION_NAME),
};

pub fn requiredExtensions() []const *align(1) const [:0]u8 {
    return &required_extensions;
}

pub fn init(allocator: Allocator, listener: Wsi.Listener) !Self {
    return Self{
        .allocator = allocator,
        .listener = listener,
    };
}

pub fn connect(self: *Self) !void {
    const hInstance = windows.GetModuleHandleW(null);

    const windowClass: windows.WNDCLASSEXW = .{
        .cbSize = @sizeOf(windows.WNDCLASSEXW), // cbSize
        .style = windows.CS_OWNDC, // | CS_HREDRAW | CS_VREDRAW*/, // style -- some window behavior
        .lpfnWndProc = wndProc, // lpfnWndProc -- set event handler
        .cbClsExtra = 0, // cbClsExtra -- set 0 extra bytes after class
        .cbWndExtra = @sizeOf(*Self), // cbWndExtra -- set 0 extra bytes after class instance
        .hInstance = hInstance, // hInstance
        .hIcon = windows.LoadIconA(null, windows.IDI_APPLICATION), // hIcon -- application icon
        .hCursor = null, // hCursor -- cursor inside
        .hbrBackground = null, //(HBRUSH)( COLOR_WINDOW + 1 ), // hbrBackground
        .lpszMenuName = null, // lpszMenuName -- menu class name
        .lpszClassName = W("vkwc"), // lpszClassName -- window class name/identificator
        .hIconSm = windows.LoadIconA(null, windows.IDI_APPLICATION), // hIconSm
    };

    // register window class
    const classAtom = windows.RegisterClassExW(&windowClass);
    if (classAtom == 0) {
        return error.FailedToRegisterWindow;
    }

    const windowedStyle = windows.WS_OVERLAPPEDWINDOW | windows.WS_CLIPCHILDREN | windows.WS_CLIPSIBLINGS;
    const windowedExStyle = windows.WS_EX_OVERLAPPEDWINDOW;

    var windowRect = windows.RECT{
        .left = 0,
        .top = 0,
        .right = 480,
        .bottom = 480,
    };
    if (windows.AdjustWindowRectEx(&windowRect, windowedStyle, windows.FALSE, windowedExStyle) == 0) {
        // throw string( "Trouble adjusting window size: " ) + to_string( GetLastError() );
        return error.FailedToAdjustWindowSize;
    }
    const hWnd = windows.CreateWindowExA(
        windowedExStyle,
        windows.MAKEINTATOM(classAtom),
        "vulkan",
        windowedStyle,
        windows.CW_USEDEFAULT,
        windows.CW_USEDEFAULT,
        windowRect.right - windowRect.left,
        windowRect.bottom - windowRect.top,
        null,
        null,
        hInstance,
        null,
    );

    if (hWnd == 0) {
        return error.FailedToCreateWindow;
    }

    windows.SetLastError(0);
    _ = windows.SetWindowLongPtr(hWnd, windows.GWLP_USERDATA, @intCast(@intFromPtr(self)));
    if (windows.GetLastError() != 0) {
        unreachable;
    }

    self.hWnd = hWnd;
    self.hInstance = hInstance;

    _ = windows.ShowWindow(hWnd, windows.SW_SHOW); // TODO

    _ = windows.SetForegroundWindow(hWnd); // TODO

    _ = windows.SetCursor(null);

    const rid = windows.RAWINPUTDEVICE{
        .usUsagePage = 0x01, // HID_USAGE_PAGE_GENERIC
        .dwFlags = windows.RIDEV_INPUTSINK,
        .usUsage = 0x02, // HID_USAGE_GENERIC_MOUSE
        .hwndTarget = hWnd,
    };

    _ = windows.RegisterRawInputDevices(&rid, 1, @sizeOf(windows.RAWINPUTDEVICE));
}

pub fn deinit(self: *Self) void {
    _ = self;
}

pub fn createVulkanSurface(self: *const Self, vkInstance: windows.VkInstance) anyerror!windows.VkSurfaceKHR {
    var createInfo = windows.VkWin32SurfaceCreateInfoKHR{
        .sType = windows.VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
        .hinstance = self.hInstance,
        .hwnd = self.hWnd,
    };

    var vulkanSurface: windows.VkSurfaceKHR = null;
    if (windows.VK_SUCCESS != windows.vkCreateWin32SurfaceKHR(vkInstance, &createInfo, null, &vulkanSurface)) return error.FailedToCreateVulkanSurface;
    return vulkanSurface;
}

pub fn dispatch(self: *Self) anyerror!void {
    _ = self;
    var msg: windows.MSG = undefined;

    while (true) {
        const ret = windows.GetMessageW(
            &msg,
            null,
            0,
            0,
        );

        if (ret == 0) {
            return;
        }

        // TODO ERROR CHECK
        _ = windows.TranslateMessage(&msg);
        _ = windows.DispatchMessageW(&msg); //dispatch to wndProc; ignore return from wndProc
    }
}

fn wndProc(hWnd: windows.HWND, uMsg: windows.UINT, wParam: windows.WPARAM, lParam: windows.LPARAM) callconv(.c) windows.LRESULT {
    const ptr = windows.GetWindowLongPtr(hWnd, windows.GWLP_USERDATA);
    if (ptr == 0) {
        return windows.DefWindowProc(hWnd, uMsg, wParam, lParam);
    }

    const self: *Self = @ptrFromInt(@as(usize, @intCast(ptr))); // TODO: check not null
    switch (uMsg) {
        windows.WM_CLOSE => {
            windows.PostQuitMessage(0);
            self.listener.vtable.close(self.listener.ptr);
            return 0;
        },
        // background will be cleared by Vulkan in WM_PAINT instead
        windows.WM_ERASEBKGND => {
            return 0;
        },

        windows.WM_MOVE => {
            _ = windows.GetWindowRect(self.hWnd, &self.wRect);
            _ = windows.ClipCursor(&self.wRect);
            return 0;
        },

        windows.WM_SIZE => {
            const width: i16 = @truncate(@as(i64, lParam) & 0xFFFF);
            const height: i16 = @truncate(@as(i64, lParam >> 16) & 0xFFFF);
            self.listener.vtable.invalidateSurface(self.listener.ptr, width, height);
            _ = windows.GetWindowRect(self.hWnd, &self.wRect);
            _ = windows.ClipCursor(&self.wRect);
            return 0;
        },

        windows.WM_PAINT => {
            self.listener.vtable.draw(self.listener.ptr);
            return 0;
        },

        windows.WM_KEYUP => {
            const key = switch (wParam) {
                'W' => Wsi.Key{ .Char = 'W' },
                'S' => Wsi.Key{ .Char = 'S' },
                'D' => Wsi.Key{ .Char = 'D' },
                'A' => Wsi.Key{ .Char = 'A' },
                windows.VK_SPACE => Wsi.Key.Space,
                windows.VK_SHIFT => Wsi.Key.Shift,
                else => {
                    return windows.DefWindowProc(hWnd, uMsg, wParam, lParam);
                },
            };
            self.listener.vtable.keyboardEvent(
                self.listener.ptr,
                .{ .KeyUp = key },
            );
            return 0;
        },
        windows.WM_KEYDOWN => {
            const keyFlags = HIWORD(@intCast(lParam));
            const wasKeyDown = (keyFlags & windows.KF_REPEAT) == windows.KF_REPEAT;
            if (!wasKeyDown) {
                const key: Wsi.Key = switch (wParam) {
                    windows.VK_ESCAPE => {
                        windows.PostQuitMessage(0);
                        self.listener.vtable.close(self.listener.ptr);
                        return 0;
                    },
                    'W' => Wsi.Key{ .Char = 'W' },
                    'S' => Wsi.Key{ .Char = 'S' },
                    'D' => Wsi.Key{ .Char = 'D' },
                    'A' => Wsi.Key{ .Char = 'A' },
                    windows.VK_SPACE => .Space,
                    windows.VK_SHIFT => .Shift,
                    else => {
                        return windows.DefWindowProc(hWnd, uMsg, wParam, lParam);
                    },
                };
                self.listener.vtable.keyboardEvent(
                    self.listener.ptr,
                    .{ .KeyDown = key },
                );
            }
            return 0;
        },

        windows.WM_INPUT => {
            if (!self.have_focus) return 0;

            var input = windows.RAWINPUT{};
            var inputSize = @as(windows.UINT, @sizeOf(@TypeOf(input)));
            _ = windows.GetRawInputData(lParam, windows.RID_INPUT, @ptrCast(&input), &inputSize, @sizeOf(windows.RAWINPUTHEADER));

            if (input.header.dwType == windows.RIM_TYPEMOUSE) {
                const dx = input.data.mouse.lLastX;
                const dy = input.data.mouse.lLastY;

                self.listener.vtable.mouseMove(
                    self.listener.ptr,
                    @floatFromInt(dx),
                    @floatFromInt(dy),
                );

                _ = windows.SetCursorPos(
                    self.wRect.left + @divTrunc(self.wRect.right - self.wRect.left, 2),
                    self.wRect.top + @divTrunc(self.wRect.bottom - self.wRect.top, 2),
                );
            }

            return 0;
        },

        windows.WM_KILLFOCUS => {
            self.have_focus = false;
            return 0;
        },

        windows.WM_SETFOCUS => {
            self.have_focus = true;
            _ = windows.SetCursor(null);
            return 0;
        },

        windows.WM_SYSCOMMAND => {
            switch (wParam) {
                windows.SC_KEYMENU => {
                    if (lParam == windows.VK_RETURN) { // Alt-Enter without "no sysmenu hotkey exists" beep
                        // toggleFullscreen(hWnd);
                        return 0;
                    } else {
                        return windows.DefWindowProc(hWnd, uMsg, wParam, lParam);
                    }
                },
                else => {
                    return windows.DefWindowProc(hWnd, uMsg, wParam, lParam);
                },
            }
        },

        else => return windows.DefWindowProc(hWnd, uMsg, wParam, lParam),
    }
}

pub inline fn GET_X_LPARAM(lparam: windows.LPARAM) i32 {
    return @intCast(@as(i16, @bitCast(@as(u16, @intCast(lparam & 0xffff)))));
}
pub inline fn GET_Y_LPARAM(lparam: windows.LPARAM) i32 {
    return @intCast(@as(i16, @bitCast(@as(u16, @intCast((lparam >> 16) & 0xffff)))));
}
pub inline fn HIWORD(dword: windows.DWORD) std.os.windows.WORD {
    return @as(std.os.windows.WORD, @bitCast(@as(u16, @intCast((dword >> 16) & 0xffff))));
}
