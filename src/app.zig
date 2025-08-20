const std = @import("std");
const builtin = @import("builtin");

const log = std.log.scoped(.app);

const mem = std.mem;
const posix = std.posix;

const Allocator = std.mem.Allocator;
const Renderer = @import("renderer.zig");
const Chunk = @import("chunk.zig");
const TextureManager = @import("texture_manager.zig");

const Wsi = switch (builtin.target.os.tag) {
    .linux => @import("wsi/wayland/wayland.zig"),
    .windows => @import("wsi/windows/windows.zig"),
    else => @panic("no wsi implemented for this platform"),
};

allocator: Allocator,
vulkanInstance: Renderer.Instance,
running: bool,
width: i32 = 0,
height: i32 = 0,
renderer: ?Renderer = null,
lastFrame: std.time.Instant,
camera: Renderer.Camera = .{},
world: [][10][10]Chunk,
textureManager: TextureManager,
wsi: Wsi,
keyboard_state: struct {
    forward: i32 = 0,
    right: i32 = 0,
    up: i32 = 0,
} = .{},

const Self = @This();

pub fn init(allocator: Allocator) !Self {
    const vulkanInstance = try Renderer.Instance.init(allocator, Wsi.requiredExtensions());

    const world = try allocator.alloc([10][10]Chunk, 10);

    var size: usize = 0;

    for (world) |*zchunk| {
        for (zchunk) |*ychunk| {
            for (ychunk) |*c| {
                c.* = Chunk.init(allocator);

                for (0..64) |x| {
                    for (0..64) |z| {
                        try c.putBlock(x, 0, z, if (std.crypto.random.boolean()) 1 else 2);
                    }
                }

                try c.putBlock(1, 2, 1, 3);
                c.compact();

                size += c.size();
            }
        }
    }

    std.log.info("World size: {}", .{size * @sizeOf(u64)});

    const textureManager = try TextureManager.init(allocator);

    const self = Self{
        .allocator = allocator,
        .vulkanInstance = vulkanInstance,
        .running = true,
        .lastFrame = try std.time.Instant.now(),
        .world = world,
        .textureManager = textureManager,
        .wsi = undefined,
    };

    return self;
}

fn newListener(self: *Self) Wsi.Listener {
    return Wsi.Listener{
        .ptr = self,
        .vtable = .{
            .close = close,
            .draw = drawListener,
            .invalidateSurface = invalidateSurface,
            .keyboardEvent = keyboardEvent,
            .mouseMove = mouseMove,
        },
    };
}

fn close(ptr: *anyopaque) void {
    const self = @as(*Self, @ptrCast(@alignCast(ptr)));
    self.running = false;
}

fn drawListener(ptr: *anyopaque) void {
    const self = @as(*Self, @ptrCast(@alignCast(ptr)));
    if (self.renderer != null) {
        self.draw() catch |e| {
            if (e == error.RecreateSwapchain) {
                log.debug("error drawing : {}, recreating swapchain", .{e});
                self.renderer.?.recreate(self.width, self.height) catch |e2| {
                    log.err("failed to recreate renderer : {}", .{e2});
                };
            } else {
                log.err("while drawing: {}", .{e});
            }
        };
    }
}

fn invalidateSurface(ptr: *anyopaque, width: i32, height: i32) void {
    const self = @as(*Self, @ptrCast(@alignCast(ptr)));
    log.debug("invalidateSurface {}x{}", .{ width, height });

    self.width = width;
    self.height = height;

    if (self.renderer) |*renderer| {
        renderer.recreate(width, height) catch |e| {
            log.err("recreate {}", .{e});
        };
    }
}

fn keyboardEvent(ptr: *anyopaque, event: Wsi.KeyboardEvent) void {
    const self = @as(*Self, @ptrCast(@alignCast(ptr)));
    switch (event) {
        .KeyDown => |key| switch (key) {
            .Char => |c| switch (c) {
                'W' => self.keyboard_state.forward += 1,
                'S' => self.keyboard_state.forward -= 1,
                'D' => self.keyboard_state.right += 1,
                'A' => self.keyboard_state.right -= 1,
                else => {},
            },
            .Space => self.keyboard_state.up += 1,
            .Shift => self.keyboard_state.up -= 1,
            else => {},
        },
        .KeyUp => |key| switch (key) {
            .Char => |c| switch (c) {
                'W' => self.keyboard_state.forward -= 1,
                'S' => self.keyboard_state.forward += 1,
                'D' => self.keyboard_state.right -= 1,
                'A' => self.keyboard_state.right += 1,
                else => {},
            },
            .Space => self.keyboard_state.up -= 1,
            .Shift => self.keyboard_state.up += 1,
            else => {},
        },
    }
}

fn mouseMove(ptr: *anyopaque, dx: f64, dy: f64) void {
    const self = @as(*Self, @ptrCast(@alignCast(ptr)));
    self.camera.yaw += @floatCast(dx * 0.1);
    self.camera.pitch += @floatCast(dy * 0.1);
}

pub fn connect(self: *Self) !void {
    const wsi = try Wsi.init(self.allocator, self.newListener());
    self.wsi = wsi;
    try self.wsi.connect();

    const renderer = try Renderer.new(Wsi, self.vulkanInstance, self.allocator, self.textureManager, self.wsi);
    self.renderer = renderer;
    self.renderer.?.updateWorld(self.world);

    try self.renderer.?.recreate(self.width, self.height);
    try self.draw();
}

pub fn deinit(self: *Self) void {
    if (self.renderer) |*r| {
        r.deinit() catch |e| {
            log.err("failed to deinit renderer: {}", .{e});
        };
    }

    self.wsi.deinit();

    for (0..10) |cz| {
        for (0..10) |cy| {
            for (0..10) |cx| {
                self.world[cz][cy][cx].deinit();
            }
        }
    }

    self.allocator.free(self.world);
}

const LOS = 2;

pub fn dispatch(self: *Self) !void {
    try self.wsi.dispatch();
}

fn draw(self: *Self) !void {
    const lastFrame = self.lastFrame;
    self.lastFrame = try std.time.Instant.now();

    const deltaTime: f32 = @floatCast(@as(f64, @floatFromInt(self.lastFrame.since(lastFrame))) / 1000000000.0);

    const camera = &self.camera;
    const keyboard_state = &self.keyboard_state;

    const yaw = std.math.degreesToRadians(camera.yaw);
    const forward = @as(f32, @floatFromInt(keyboard_state.forward));
    const right = @as(f32, @floatFromInt(keyboard_state.right));

    const x = std.math.sin(yaw) * forward + std.math.cos(-yaw) * right;
    const z = std.math.cos(yaw) * forward + std.math.sin(-yaw) * right;

    camera.x += @as(f32, x) * deltaTime;
    camera.y += @as(f32, @floatFromInt(keyboard_state.up)) * deltaTime;
    camera.z += @as(f32, z) * deltaTime;

    try self.renderer.?.draw(&self.camera);
}
