const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

const log = std.log.scoped(.wayland);

const vulkan = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
});

const wayland = @import("wayland");
const wl = wayland.client.wl;
const xdg = wayland.client.xdg;
const zxdg = wayland.client.zxdg;
const zwp = wayland.client.zwp;

const Keyboard = @import("keyboard.zig");
const Wsi = @import("../wsi.zig");

pub const Listener = Wsi.Listener;
pub const ListenerVTable = Wsi.ListenerVtable;
pub const KeyboardEvent = Wsi.KeyboardEvent;

const Context = struct {
    shm: ?*wl.Shm = null,
    compositor: ?*wl.Compositor = null,
    wm_base: ?*xdg.WmBase = null,
    display: *wl.Display = undefined,
    registry: *wl.Registry = undefined,
    surface: *wl.Surface = undefined,
    xdg_surface: *xdg.Surface = undefined,
    xdg_top_level: *xdg.Toplevel = undefined,
    zxdg_decoration_manager: ?*zxdg.DecorationManagerV1 = null,
    zxdg_decoration: ?*zxdg.ToplevelDecorationV1 = null,
    seat: *wl.Seat = undefined,
    pointer: ?*wl.Pointer = null,
    pointer_constraint: *zwp.PointerConstraintsV1 = undefined,
    locked_pointer: ?*zwp.LockedPointerV1 = null,
    relative_pointer_manager: *zwp.RelativePointerManagerV1 = undefined,
    relative_pointer: ?*zwp.RelativePointerV1 = null,
    keyboard: ?*wl.Keyboard = null,
    keyboardParser: Keyboard = undefined,
    keyboard_state: struct {
        forward: i32 = 0,
        right: i32 = 0,
        up: i32 = 0,
    } = .{},
};

const Self = @This();

allocator: Allocator,
context: Context = .{},
listener: Wsi.Listener,

const required_extensions: [1]*align(1) const [:0]u8 = .{
    @ptrCast(vulkan.VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME),
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

pub fn createVulkanSurface(self: *const Self, vkInstance: vulkan.VkInstance) !vulkan.VkSurfaceKHR {
    var createInfo = vulkan.VkWaylandSurfaceCreateInfoKHR{};
    createInfo.sType = vulkan.VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
    createInfo.display = @ptrCast(self.context.display);
    createInfo.surface = @ptrCast(self.context.surface);

    var vulkanSurface: vulkan.VkSurfaceKHR = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateWaylandSurfaceKHR(vkInstance, &createInfo, null, &vulkanSurface)) return error.FailedToCreateVulkanSurface;
    return vulkanSurface;
}

pub fn connect(self: *Self) !void {
    const display = try wl.Display.connect(null);
    const registry = try display.getRegistry();

    self.context.display = display;
    self.context.registry = registry;

    registry.setListener(*Self, registryListener, self);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    //const shm = context.shm orelse return error.NoWlShm;
    const compositor = self.context.compositor orelse return error.NoWlCompositor;
    const wm_base = self.context.wm_base orelse return error.NoXdgWmBase;

    self.context.surface = try compositor.createSurface();
    self.context.xdg_surface = try wm_base.getXdgSurface(self.context.surface);
    self.context.xdg_top_level = try self.context.xdg_surface.getToplevel();

    self.context.xdg_surface.setListener(*wl.Surface, xdgSurfaceListener, self.context.surface);
    self.context.xdg_top_level.setListener(*Self, xdgToplevelListener, self);

    if (self.context.zxdg_decoration_manager) |dec| {
        self.context.zxdg_decoration = try dec.getToplevelDecoration(self.context.xdg_top_level);
        self.context.zxdg_decoration.?.setMode(.server_side);
    }

    self.context.surface.commit();

    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;
}

pub fn deinit(self: *Self) void {
    self.context.keyboardParser.deinit();
    if (self.context.relative_pointer) |pointer| {
        pointer.destroy();
    }
    self.context.relative_pointer_manager.destroy();
    if (self.context.locked_pointer) |p| {
        p.destroy();
    }
    self.context.pointer_constraint.destroy();
    if (self.context.pointer) |pointer| {
        pointer.release();
    }
    if (self.context.keyboard) |keyboard| {
        keyboard.release();
    }
    self.context.seat.destroy();
    if (self.context.zxdg_decoration) |dec| {
        dec.destroy();
    }
    if (self.context.zxdg_decoration_manager) |manager| {
        manager.destroy();
    }
    self.context.xdg_top_level.destroy();
    self.context.xdg_surface.destroy();
    self.context.surface.destroy();
    self.context.registry.destroy();
    self.context.display.disconnect();
}

pub fn dispatch(self: *Self) !void {
    self.listener.vtable.draw(self.listener.ptr);
    if (self.context.display.dispatch() != .SUCCESS) return error.DispatchFailed;
}

fn registryListener(registry: *wl.Registry, event: wl.Registry.Event, data: *Self) void {
    var context = &data.context;
    switch (event) {
        .global => |global| {
            log.debug("registry interface : {s}", .{global.interface});
            if (mem.orderZ(u8, global.interface, wl.Compositor.interface.name) == .eq) {
                context.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, wl.Shm.interface.name) == .eq) {
                context.shm = registry.bind(global.name, wl.Shm, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, xdg.WmBase.interface.name) == .eq) {
                context.wm_base = registry.bind(global.name, xdg.WmBase, 1) catch return;
                context.wm_base.?.setListener(?*anyopaque, wmBaseListener, null);
            } else if (mem.orderZ(u8, global.interface, zxdg.DecorationManagerV1.interface.name) == .eq) {
                context.zxdg_decoration_manager = registry.bind(global.name, zxdg.DecorationManagerV1, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, wl.Seat.interface.name) == .eq) {
                context.seat = registry.bind(global.name, wl.Seat, 8) catch return;
                context.seat.setListener(*Self, wlSeatListener, data);
            } else if (mem.orderZ(u8, global.interface, zwp.PointerConstraintsV1.interface.name) == .eq) {
                context.pointer_constraint = registry.bind(global.name, zwp.PointerConstraintsV1, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, zwp.RelativePointerManagerV1.interface.name) == .eq) {
                context.relative_pointer_manager = registry.bind(global.name, zwp.RelativePointerManagerV1, 1) catch return;
            }
        },
        .global_remove => {},
    }
}

fn wmBaseListener(wm_base: *xdg.WmBase, event: xdg.WmBase.Event, _: ?*anyopaque) void {
    switch (event) {
        .ping => |ping| {
            wm_base.pong(ping.serial);
        },
    }
}

fn xdgSurfaceListener(xdg_surface: *xdg.Surface, event: xdg.Surface.Event, surface: *wl.Surface) void {
    _ = surface;
    switch (event) {
        .configure => |configure| {
            xdg_surface.ackConfigure(configure.serial);
            log.info("configure", .{});
        },
    }
}

fn xdgToplevelListener(_: *xdg.Toplevel, event: xdg.Toplevel.Event, wsi: *Self) void {
    switch (event) {
        .configure => |configure| {
            const newWidth = if (configure.width == 0) 640 else configure.width;
            const newHeight = if (configure.height == 0) 400 else configure.height;
            log.debug("xdg top level configure {}x{}", .{ newWidth, newHeight });
            wsi.listener.vtable.invalidateSurface(
                wsi.listener.ptr,
                newWidth,
                newHeight,
            );
        },
        .wm_capabilities => {},
        .configure_bounds => {},
        .close => {
            wsi.listener.vtable.close(wsi.listener.ptr);
        },
    }
}

fn wlSeatListener(seat: *wl.Seat, event: wl.Seat.Event, data: *Self) void {
    _ = seat;
    switch (event) {
        .name => |name| {
            log.debug("seat name : {s}", .{name.name});
        },
        .capabilities => |capabilities| {
            log.debug("seat capabilities : pointer = {}, keyboard = {}, touch = {}", .{
                capabilities.capabilities.pointer,
                capabilities.capabilities.keyboard,
                capabilities.capabilities.touch,
            });
            if (capabilities.capabilities.pointer and data.context.pointer == null) {
                // todo: multiple
                data.context.pointer = data.context.seat.getPointer() catch unreachable;
                data.context.pointer.?.setListener(*Self, wlPointerListener, data);
                data.context.locked_pointer = data.context.pointer_constraint.lockPointer(data.context.surface, data.context.pointer.?, null, .persistent) catch unreachable;
                data.context.relative_pointer = data.context.relative_pointer_manager.getRelativePointer(data.context.pointer.?) catch unreachable; //todo
                data.context.relative_pointer.?.setListener(*Self, relativePointerListener, data);
            }

            if (capabilities.capabilities.keyboard and data.context.keyboard == null) {
                data.context.keyboard = data.context.seat.getKeyboard() catch unreachable;
                data.context.keyboardParser = Keyboard.init() catch unreachable;
                data.context.keyboard.?.setListener(*Self, wlKeyboardListener, data);
            }
        },
    }
}

fn wlPointerListener(pointer: *wl.Pointer, event: wl.Pointer.Event, data: *Self) void {
    _ = pointer;
    switch (event) {
        .enter => |enter| {
            data.context.pointer.?.setCursor(enter.serial, null, 0, 0);
        },
        .leave => |_| {},
        .motion => {},
        .button => |_| {},
        .axis => |_| {},
        .frame => {},
        .axis_source => |_| {},
        .axis_stop => |_| {},
        .axis_discrete => |_| {},
        .axis_value120 => |_| {},
    }
}
fn relativePointerListener(relative_pointer_v1: *zwp.RelativePointerV1, event: zwp.RelativePointerV1.Event, self: *Self) void {
    _ = relative_pointer_v1;
    switch (event) {
        .relative_motion => |motion| {
            self.listener.vtable.mouseMove(
                self.listener.ptr,
                motion.dx_unaccel.toDouble(),
                motion.dy_unaccel.toDouble(),
            );
        },
    }
}
fn wlKeyboardListener(keyboard: *wl.Keyboard, event: wl.Keyboard.Event, self: *Self) void {
    _ = keyboard;
    switch (event) {
        .keymap => |keymap| {
            if (keymap.format == .xkb_v1) {
                self.context.keyboardParser.parseKeymap(keymap.fd, keymap.size) catch unreachable;
            } else {
                log.warn("unknown keymap : {}", .{keymap.format});
            }
        },
        .enter => |e| {
            const keys = e.keys.slice(u32);
            for (keys) |key| {
                const xkbKey = Keyboard.toLower(self.context.keyboardParser.parseKeyCode(key));
                const wsi_key = switch (xkbKey) {
                    Keyboard.xkbcommon.XKB_KEY_w => Wsi.Key{ .Char = 'W' },
                    Keyboard.xkbcommon.XKB_KEY_s => Wsi.Key{ .Char = 'S' },
                    Keyboard.xkbcommon.XKB_KEY_d => Wsi.Key{ .Char = 'D' },
                    Keyboard.xkbcommon.XKB_KEY_a => Wsi.Key{ .Char = 'A' },
                    Keyboard.xkbcommon.XKB_KEY_space => .Space,
                    Keyboard.xkbcommon.XKB_KEY_Shift_L => .Shift,
                    else => continue,
                };

                self.listener.vtable.keyboardEvent(
                    self.listener.ptr,
                    .{ .KeyUp = wsi_key },
                );
            }
        },
        .leave => {},
        .key => |key| {
            const xkbKey = Keyboard.toLower(self.context.keyboardParser.parseKeyCode(key.key));

            const wsi_key = switch (xkbKey) {
                Keyboard.xkbcommon.XKB_KEY_w => Wsi.Key{ .Char = 'W' },
                Keyboard.xkbcommon.XKB_KEY_s => Wsi.Key{ .Char = 'S' },
                Keyboard.xkbcommon.XKB_KEY_d => Wsi.Key{ .Char = 'D' },
                Keyboard.xkbcommon.XKB_KEY_a => Wsi.Key{ .Char = 'A' },
                Keyboard.xkbcommon.XKB_KEY_space => .Space,
                Keyboard.xkbcommon.XKB_KEY_Shift_L => .Shift,
                else => return,
            };

            self.listener.vtable.keyboardEvent(
                self.listener.ptr,
                if (key.state == .released) .{ .KeyUp = wsi_key } else .{ .KeyDown = wsi_key },
            );
        },
        .modifiers => |modifiers| {
            self.context.keyboardParser.setModifier(modifiers.mods_depressed, modifiers.mods_latched, modifiers.mods_locked, modifiers.group);
        },
        .repeat_info => {},
    }
}
