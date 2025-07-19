const std = @import("std");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;

const App = @import("app.zig");
const Renderer = @import("renderer.zig");

var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

pub fn main() !void {
    const allocator, const is_debug = gpa: {
        if (builtin.os.tag == .wasi) break :gpa .{ std.heap.wasm_allocator, false };
        break :gpa switch (builtin.mode) {
            .Debug, .ReleaseSafe => .{ debug_allocator.allocator(), true },
            .ReleaseFast, .ReleaseSmall => .{ std.heap.smp_allocator, false },
        };
    };
    defer if (is_debug) {
        _ = debug_allocator.deinit();
    };

    var app = try App.init(allocator);
    defer app.deinit();
    try app.connect();

    while (app.running) {
        try app.dispatch();
    }

    std.log.info("end", .{});
}
